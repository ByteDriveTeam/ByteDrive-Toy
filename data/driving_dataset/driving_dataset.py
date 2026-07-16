"""驾驶模型双帧数据集：产当前/上一帧输入、帧间刚性变换、道路线图与驾驶多任务监督。

模块: data/driving_dataset/driving_dataset.py
依赖: torch, numpy, config.schema.Config, data.single_frame_base.SingleFrameSceneBase,
      data.driving_targets, data.hd_map.HdMap, vis.data_vis.geometry, data.driving_dataset.checks.*
读取配置:
    data.driving.scene_root / camera / map_dir / map_name_template / previous_frame_offset /
        dist_sigma_m / lane_half_width_m
    data.driving.lane_map.line_width_m / type_to_class / unknown_class
    data.driving.traffic_control.route_corridor_m / line_expand_m / actor_match_radius_m / stop_margin_m /
        reaction_time_s / comfortable_decel_mps2
    data.driving.box_min_visible_pixels
    data.driving.target_min_m / target_max_m（目标点采样距离窗口）
    data.driving.behavior.stationary_speed_mps / acceleration_threshold_mps2 / turn_angle_deg /
        traffic_light_semantic_tag / traffic_light_match_radius_m / traffic_light_seg_margin_px /
        traffic_light_min_pixels
    data.dataset.dino_mean / dino_std
    model.driving.bev.x_min_m / x_max_m / y_min_m / y_max_m / fov_deg
    model.driving.fields.up_channels（推导场分辨率 = bev.height/width · 2^L）
    model.driving.trajectory.num_waypoints / num_modes
    model.driving.traffic_control.state_names
    model.physics.depth_max_m（风险场包络排除超范围/天空像素）
对外接口:
    - DrivingDataset(cfg) -> torch.utils.data.Dataset
        __getitem__(i) -> dict[str, Tensor]
说明: 复用 SingleFrameSceneBase 的索引/reader 缓存/RGB 归一化。每个样本同时返回同场景上一帧 RGB 及把
      上一帧 ego 平面坐标变到当前 ego 系的 3×3 刚性矩阵；场景开头返回当前 RGB、identity 与 previous_valid=0。
      轨迹 GT 由同场景未来 num_waypoints 帧 ego 世界位姿经 world_to_ego 变到当前 ego 系；行为 GT 为固定八类
      多热向量，组合当前速度/帧间加速度、未来轨迹、动态 Agent 框与路线相关交通灯状态；红灯停车在接近阶段即激活。
      目标点沿未来自车轨迹搜距当前 target_min~target_max m 的点随机取一（近端引导 + 鲁棒），变到 ego 系。
      风险场由 GT 深度反投影包络；可行驶场先由 HD 地图按位姿栅格化，再扣除由 GT 深度确认可见的
      vehicle/pedestrian box 占用（运动类别间不分类，ego/静态环境框排除），并转成道路外/占用距离场供轨迹约束使用；
      独立道路线图由 HD Map 的 Type 与每点 yaw 栅格化为类别和有向单位切向量；分布场由 GT 航点高斯软化，视场掩码为常量
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
from vis.data_vis.geometry import transform_matrix, transform_points, world_to_ego


__all__ = ["DrivingDataset"]


class DrivingDataset(SingleFrameSceneBase):
    """以当前帧为索引、同时读取上一帧的双帧驾驶数据集。"""

    def __init__(self, cfg: Config) -> None:
        drv_data = cfg.data.driving
        super().__init__(drv_data.scene_root, drv_data.camera,
                         cfg.data.dataset.dino_mean, cfg.data.dataset.dino_std)
        self._cfg_data = drv_data
        bev = cfg.model.driving.bev
        self._fov = bev.fov_deg
        self._previous_offset = drv_data.previous_frame_offset
        # 场分辨率 = BEV 工作分辨率 · 上采样倍率（与 field_decoder 输出一致）
        scale = 2 ** len(cfg.model.driving.fields.up_channels)
        self._bev = dt.BevParams(bev.x_min_m, bev.x_max_m, bev.y_min_m, bev.y_max_m,
                                 bev.height * scale, bev.width * scale)
        self._num_waypoints = cfg.model.driving.trajectory.num_waypoints
        self._num_modes = cfg.model.driving.trajectory.num_modes
        self._depth_max_m = cfg.model.physics.depth_max_m  # 风险场包络排除超范围/天空像素
        self._box_min_visible_pixels = drv_data.box_min_visible_pixels
        self._target_min = drv_data.target_min_m
        self._target_max = drv_data.target_max_m
        self._traffic_cfg = drv_data.traffic_control
        self._traffic_state_names = cfg.model.driving.traffic_control.state_names
        behavior = drv_data.behavior
        self._behavior_params = dt.BehaviorParams(
            behavior.stationary_speed_mps, behavior.acceleration_threshold_mps2,
            behavior.turn_angle_deg, drv_data.lane_half_width_m,
            behavior.traffic_light_semantic_tag, behavior.traffic_light_match_radius_m,
            behavior.traffic_light_seg_margin_px, behavior.traffic_light_min_pixels)
        self._map_dir = resolve_repo_path(drv_data.map_dir)
        self._inview_np = dt.inview_mask(self._bev, self._fov)
        self._inview = torch.from_numpy(self._inview_np)  # 常量，预算一次
        self._hd_maps: Dict[str, HdMap] = {}
        self._state_cache: Dict[str, tuple] = {}  # 每场景 (ego 位姿 [F,6], 标量速度加速度 [F])

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        scene_dir, frame_idx = self.frame_index[i]
        reader = self.reader(scene_dir)
        meta = reader.meta
        cam = self._camera
        check_camera_calib(meta, cam)

        previous_idx = max(frame_idx - self._previous_offset, 0)
        previous_valid = float(frame_idx >= self._previous_offset)
        previous_meta = reader.frame_meta(previous_idx) if previous_valid else None
        previous_rgb = reader.rgb(previous_idx, cam) if previous_valid else None

        frame = reader.frame(frame_idx)
        check_behavior_annotations(meta, frame, cam)
        intr = meta["intrinsics"][cam]
        extrinsic6 = [float(v) for v in meta["extrinsics"][cam]]
        intrinsics4 = [float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"])]

        pose = [float(v) for v in frame["ego"]["transform"]]
        world_vel = np.array(frame["ego"]["velocity"], dtype=np.float64)
        previous_meta = previous_meta or frame["meta"]
        previous_rgb = previous_rgb if previous_rgb is not None else frame["rgb"][cam]
        previous_pose = [float(v) for v in previous_meta["ego"]["transform"]]
        previous_to_current = _planar_previous_to_current(previous_pose, pose)

        poses, accelerations = self._scene_states(scene_dir, reader)  # 全帧位姿/加速度（缓存）
        waypoints, valid = self._trajectory(poses, frame_idx, pose)
        sector = dt.sector_of(waypoints, valid, self._fov, self._num_modes)
        target_point = self._target_point(poses, frame_idx, pose, meta)
        hd_map = self._hd_map(meta["map"])
        speed_mps = float(np.linalg.norm(world_vel[:2]))
        traffic = self._traffic_targets(
            hd_map, poses, frame_idx, pose, target_point, meta, frame, speed_mps)
        behavior = dt.behavior_targets(
            waypoints, valid, speed_mps, float(accelerations[frame_idx]),
            frame["bboxes"], meta["traffic_lights"], frame["traffic_light_states"],
            meta["static_bboxes"], frame["semantic"][cam], pose, intr, extrinsic6,
            self._bev, self._fov, self._behavior_params,
            red_light_relevant=bool(traffic["red_stop_valid"]))

        depth = np.ascontiguousarray(frame["depth"][cam]).astype(np.float32)
        risk = dt.risk_field(depth, intrinsics4, extrinsic6, self._bev, self._fov, self._depth_max_m)
        map_drivable = hd_map.drivable_bev(
            pose, self._bev, self._cfg_data.lane_half_width_m)
        lane_cfg = self._cfg_data.lane_map
        lane_class, lane_direction = hd_map.lane_map_bev(
            pose, self._bev, lane_cfg.line_width_m,
            lane_cfg.type_to_class, lane_cfg.unknown_class)
        box_occupancy = dt.visible_moving_box_occupancy(
            frame["bboxes"], depth, intr, pose, extrinsic6,
            self._bev, self._depth_max_m, self._box_min_visible_pixels)
        drivable = map_drivable * (1.0 - box_occupancy)
        offroad_distance = offroad_distance_field(drivable, self._bev)
        distribution = dt.distribution_field(waypoints, valid, self._bev, self._cfg_data.dist_sigma_m)

        sample = {
            "rgb": self.normalize_rgb(frame["rgb"][cam]),
            "previous_rgb": self.normalize_rgb(previous_rgb),
            "previous_to_current": torch.from_numpy(previous_to_current),
            "previous_valid": torch.tensor(previous_valid, dtype=torch.float32),
            "intrinsics": torch.tensor(intrinsics4, dtype=torch.float32),
            "extrinsics": torch.tensor(extrinsic6, dtype=torch.float32),
            "target_point": torch.tensor(target_point, dtype=torch.float32),
            "trajectory": torch.from_numpy(waypoints),
            "traj_valid": torch.from_numpy(valid),
            "sector": torch.tensor(sector, dtype=torch.long),
            "behavior": torch.from_numpy(behavior),
            "risk": torch.from_numpy(risk),
            "drivable": torch.from_numpy(drivable),
            "lane_class": torch.from_numpy(lane_class),
            "lane_direction": torch.from_numpy(lane_direction),
            "offroad_distance": torch.from_numpy(offroad_distance),
            "distribution": torch.from_numpy(distribution),
            "inview": self._inview,
        }
        sample.update({name: torch.from_numpy(value) if isinstance(value, np.ndarray)
                       else torch.tensor(value, dtype=torch.float32)
                       for name, value in traffic.items()})
        return sample

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

    def _route_polyline(self, poses, frame_idx, pose, target_point):
        """截取未来专家路径的目标距离窗口；长时间等灯时仍能延伸到路口之后。"""
        future = poses[frame_idx + 1:, :3]
        future_ego = transform_points(future, world_to_ego(pose))[:, :2].astype(np.float32)
        route = np.vstack((np.zeros((1, 2), dtype=np.float32), future_ego))
        arclength = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]
        end = min(int(np.searchsorted(arclength, self._target_max, side="right")) + 1, len(route))
        route = route[:end]
        if len(route) < 2 or np.linalg.norm(target_point) > np.linalg.norm(route[-1]) + 1e-3:
            route = np.vstack((route, target_point))
        return route

    def _traffic_targets(self, hd_map, poses, frame_idx, pose, target_point, meta, frame, speed_mps):
        """生成路线相关交通控制监督，并用视场与舒适制动距离门控红灯停车约束。"""
        traffic = hd_map.traffic_control_bev(
            pose, self._route_polyline(poses, frame_idx, pose, target_point),
            meta["traffic_lights"], frame["traffic_light_states"], self._bev,
            self._traffic_cfg.route_corridor_m, self._traffic_cfg.line_expand_m,
            self._traffic_cfg.actor_match_radius_m, self._traffic_state_names)
        stopping_distance = (speed_mps * self._traffic_cfg.reaction_time_s
                             + speed_mps ** 2 / (2.0 * self._traffic_cfg.comfortable_decel_mps2)
                             + self._traffic_cfg.stop_margin_m)
        line_inview = bool(np.any(traffic["stop_line"] * self._inview_np))
        can_stop = (speed_mps <= self._behavior_params.stationary_speed_mps
                    or float(traffic["stop_distance"]) >= stopping_distance)
        traffic["red_stop_valid"] = np.float32(
            bool(traffic["red_stop_valid"]) and line_inview and can_stop)
        return traffic

    def _hd_map(self, map_name: str) -> HdMap:
        """按场景 map 名（去 _Opt 后缀）惰性加载并缓存 HD 地图。"""
        key = map_name.replace("_Opt", "")
        if key not in self._hd_maps:
            path = self._map_dir / self._cfg_data.map_name_template.format(map=key)
            self._hd_maps[key] = HdMap(path)
        return self._hd_maps[key]


def _planar_previous_to_current(previous_pose, current_pose):
    """由两帧世界位姿提取上一帧 ego xy → 当前帧 ego xy 的齐次刚性矩阵。"""
    transform = world_to_ego(current_pose) @ transform_matrix(previous_pose)
    return np.array([
        [transform[0, 0], transform[0, 1], transform[0, 3]],
        [transform[1, 0], transform[1, 1], transform[1, 3]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
