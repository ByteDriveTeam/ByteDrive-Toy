"""驾驶模型单帧数据集：逐帧产模型输入、三场/轨迹/行为 GT 及 HDMap 越界距离场。

模块: data/driving_dataset/driving_dataset.py
依赖: torch, numpy, config.schema.Config, data.single_frame_base.SingleFrameSceneBase,
      data.driving_targets, data.hd_map.HdMap, vis.data_vis.geometry, data.driving_dataset.checks.*
读取配置:
    data.driving.scene_root / camera / map_dir / map_name_template / dist_sigma_m / lane_half_width_m
    data.driving.target_min_m / target_max_m（目标点采样距离窗口）
    data.driving.behavior.stationary_speed_mps / acceleration_threshold_mps2 / turn_angle_deg /
        traffic_light_semantic_tag / traffic_light_match_radius_m / traffic_light_seg_margin_px /
        traffic_light_min_pixels
    data.dataset.dino_mean / dino_std
    model.driving.bev.x_min_m / x_max_m / y_min_m / y_max_m / fov_deg
    model.driving.fields.up_channels（推导场分辨率 = bev.height/width · 2^L）
    model.driving.trajectory.num_waypoints / num_modes
    model.physics.depth_max_m（风险场包络排除超范围/天空像素）
对外接口:
    - DrivingDataset(cfg) -> torch.utils.data.Dataset
        __getitem__(i) -> dict[str, Tensor]
说明: 复用 SingleFrameSceneBase 的索引/reader 缓存/RGB 归一化。轨迹 GT 由同场景未来 num_waypoints 帧 ego 世界
      位姿经 world_to_ego 变到当前 ego 系（帧间隔即采集节拍）；自车速度由世界速度旋转到 ego 系（前向 vx、右向
      vy）；行为 GT 为固定八类多热向量，组合当前速度/帧间加速度、未来轨迹、动态 Agent 框、交通灯状态与 Seg
      可见性判定。目标点沿未来自车轨迹搜距当前 target_min~target_max m 的点随机取一（近端引导 + 鲁棒），变到 ego 系。
      风险场由 GT 深度反投影包络，可行驶场由 HD 地图按位姿栅格化，并转成道路外距离场供轨迹越界损失使用；
      分布场由 GT 航点高斯软化，视场掩码为常量
      （构造期预算）。全帧 ego 位姿与速度加速度按场景缓存，供轨迹/行为/目标点复用、避免逐样本重复读 LMDB。场分辨率与
      模型上采样输出一致（Hb·2^L）。HD 地图按场景 map 名（去 _Opt 后缀）惰性加载并缓存。几何投影复用
      vis.data_vis.geometry / data.driving_targets。
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from config.schema import Config
from data import driving_targets as dt
from data.driving_dataset.checks.driving_dataset_checks import check_behavior_annotations, check_camera_calib
from data.hd_map import HdMap, offroad_distance_field
from data.single_frame_base import SingleFrameSceneBase, resolve_repo_path
from vis.data_vis.geometry import transform_points, world_to_ego


__all__ = ["DrivingDataset"]


class DrivingDataset(SingleFrameSceneBase):
    """逐帧展开的单帧驾驶数据集（复用感知单帧读取，另产驾驶多任务 GT）。"""

    def __init__(self, cfg: Config) -> None:
        drv_data = cfg.data.driving
        super().__init__(drv_data.scene_root, drv_data.camera,
                         cfg.data.dataset.dino_mean, cfg.data.dataset.dino_std)
        self._cfg_data = drv_data
        bev = cfg.model.driving.bev
        self._fov = bev.fov_deg
        # 场分辨率 = BEV 工作分辨率 · 上采样倍率（与 field_decoder 输出一致）
        scale = 2 ** len(cfg.model.driving.fields.up_channels)
        self._bev = dt.BevParams(bev.x_min_m, bev.x_max_m, bev.y_min_m, bev.y_max_m,
                                 bev.height * scale, bev.width * scale)
        self._num_waypoints = cfg.model.driving.trajectory.num_waypoints
        self._num_modes = cfg.model.driving.trajectory.num_modes
        self._depth_max_m = cfg.model.physics.depth_max_m  # 风险场包络排除超范围/天空像素
        self._target_min = drv_data.target_min_m
        self._target_max = drv_data.target_max_m
        behavior = drv_data.behavior
        self._behavior_params = dt.BehaviorParams(
            behavior.stationary_speed_mps, behavior.acceleration_threshold_mps2,
            behavior.turn_angle_deg, drv_data.lane_half_width_m,
            behavior.traffic_light_semantic_tag, behavior.traffic_light_match_radius_m,
            behavior.traffic_light_seg_margin_px, behavior.traffic_light_min_pixels)
        self._map_dir = resolve_repo_path(drv_data.map_dir)
        self._inview = torch.from_numpy(dt.inview_mask(self._bev, self._fov))  # 常量，预算一次
        self._hd_maps: Dict[str, HdMap] = {}
        self._state_cache: Dict[str, tuple] = {}  # 每场景 (ego 位姿 [F,6], 标量速度加速度 [F])

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        scene_dir, frame_idx = self.frame_index[i]
        reader = self.reader(scene_dir)
        meta = reader.meta
        cam = self._camera
        check_camera_calib(meta, cam)

        frame = reader.frame(frame_idx)
        check_behavior_annotations(meta, frame, cam)
        intr = meta["intrinsics"][cam]
        extrinsic6 = [float(v) for v in meta["extrinsics"][cam]]
        intrinsics4 = [float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"])]

        pose = [float(v) for v in frame["ego"]["transform"]]
        world_vel = np.array(frame["ego"]["velocity"], dtype=np.float64)
        ego_vel = (world_to_ego(pose)[:3, :3] @ world_vel)[:2]        # 世界速度 → ego 系 (前, 右)

        poses, accelerations = self._scene_states(scene_dir, reader)  # 全帧位姿/加速度（缓存）
        waypoints, valid = self._trajectory(poses, frame_idx, pose)
        sector = dt.sector_of(waypoints, valid, self._fov, self._num_modes)
        target_point = self._target_point(poses, frame_idx, pose, meta)
        behavior = dt.behavior_targets(
            waypoints, valid, float(np.linalg.norm(world_vel[:2])), float(accelerations[frame_idx]),
            frame["bboxes"], meta["traffic_lights"], frame["traffic_light_states"],
            meta["static_bboxes"], frame["semantic"][cam], pose, intr, extrinsic6,
            self._bev, self._fov, self._behavior_params)

        depth = np.ascontiguousarray(frame["depth"][cam]).astype(np.float32)
        risk = dt.risk_field(depth, intrinsics4, extrinsic6, self._bev, self._fov, self._depth_max_m)
        drivable = self._hd_map(meta["map"]).drivable_bev(pose, self._bev, self._cfg_data.lane_half_width_m)
        offroad_distance = offroad_distance_field(drivable, self._bev)
        distribution = dt.distribution_field(waypoints, valid, self._bev, self._cfg_data.dist_sigma_m)

        return {
            "rgb": self.normalize_rgb(frame["rgb"][cam]),
            "intrinsics": torch.tensor(intrinsics4, dtype=torch.float32),
            "extrinsics": torch.tensor(extrinsic6, dtype=torch.float32),
            "ego_velocity": torch.tensor(ego_vel, dtype=torch.float32),
            "target_point": torch.tensor(target_point, dtype=torch.float32),
            "trajectory": torch.from_numpy(waypoints),
            "traj_valid": torch.from_numpy(valid),
            "sector": torch.tensor(sector, dtype=torch.long),
            "behavior": torch.from_numpy(behavior),
            "risk": torch.from_numpy(risk),
            "drivable": torch.from_numpy(drivable),
            "offroad_distance": torch.from_numpy(offroad_distance),
            "distribution": torch.from_numpy(distribution),
            "inview": self._inview,
        }

    def _scene_states(self, scene_dir, reader):
        """惰性缓存全帧 ego 位姿与速度加速度（轻量读 LMDB），供轨迹/行为/目标点复用。"""
        key = str(scene_dir)
        if key not in self._state_cache:
            metas = [reader.frame_meta(j) for j in range(reader.num_frames)]
            poses = np.array([meta["ego"]["transform"] for meta in metas], dtype=np.float64)
            velocities = np.array([meta["ego"]["velocity"] for meta in metas], dtype=np.float64)
            sim_times = np.array([meta["sim_time"] for meta in metas], dtype=np.float64)
            self._state_cache[key] = (poses, dt.speed_accelerations(velocities, sim_times))
        return self._state_cache[key]

    def _trajectory(self, poses: np.ndarray, frame_idx: int, pose):
        """取同场景未来 num_waypoints 帧 ego 世界位姿，变到当前 ego 系得航点与有效掩码。"""
        future = list(poses[frame_idx + 1: frame_idx + 1 + self._num_waypoints])
        return dt.trajectory_targets(future, pose, self._num_waypoints)

    def _target_point(self, poses: np.ndarray, frame_idx: int, pose, meta):
        """沿未来自车轨迹搜距当前 [target_min, target_max]m 的点随机取一作近端导航目标（变到当前 ego 系）。

        近端引导比「整条路线终点」更明确，且窗口内随机选点增强对目标位置扰动的鲁棒性。无点落入窗口（临近场景
        末尾/慢行）时取最远未来点；无未来帧（场景末帧）时退回路线终点。
        """
        future = poses[frame_idx + 1:]                               # [m,6]
        if len(future) == 0:
            end = np.array([meta["route"]["end"][:3]], dtype=np.float64)
            return transform_points(end, world_to_ego(pose))[0, :2].astype(np.float32)
        dist = np.hypot(future[:, 0] - pose[0], future[:, 1] - pose[1])
        within = np.nonzero((dist >= self._target_min) & (dist <= self._target_max))[0]
        j = int(np.random.choice(within)) if len(within) > 0 else int(np.argmax(dist))
        ego_pt = transform_points(future[j:j + 1, :3], world_to_ego(pose))
        return ego_pt[0, :2].astype(np.float32)

    def _hd_map(self, map_name: str) -> HdMap:
        """按场景 map 名（去 _Opt 后缀）惰性加载并缓存 HD 地图。"""
        key = map_name.replace("_Opt", "")
        if key not in self._hd_maps:
            path = self._map_dir / self._cfg_data.map_name_template.format(map=key)
            self._hd_maps[key] = HdMap(path)
        return self._hd_maps[key]
