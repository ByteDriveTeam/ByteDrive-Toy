"""驾驶模型单帧数据集：逐帧产 RGB/内外参/自车速度/目标点，及轨迹/风险/可行驶/分布场与视场 GT。

模块: data/driving_dataset/driving_dataset.py
依赖: torch, numpy, config.schema.Config, data.single_frame_base.SingleFrameSceneBase,
      data.driving_targets, data.hd_map.HdMap, vis.data_vis.geometry, data.driving_dataset.checks.*
读取配置:
    data.driving.scene_root / camera / map_dir / map_name_template / dist_sigma_m / lane_half_width_m
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
      vy）；目标点取路线终点变到 ego 系作导航意图。风险场由 GT 深度反投影，可行驶场由 HD 地图按位姿栅格化，
      分布场由 GT 航点高斯软化，视场掩码为常量（构造期预算）。场分辨率与模型上采样输出一致（Hb·2^L）。HD 地图
      按场景 map 名（去 _Opt 后缀）惰性加载并缓存。几何投影全部复用 vis.data_vis.geometry / data.driving_targets。
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from config.schema import Config
from data import driving_targets as dt
from data.driving_dataset.checks.driving_dataset_checks import check_camera_calib
from data.hd_map import HdMap
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
        self._map_dir = resolve_repo_path(drv_data.map_dir)
        self._inview = torch.from_numpy(dt.inview_mask(self._bev, self._fov))  # 常量，预算一次
        self._hd_maps: Dict[str, HdMap] = {}

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        scene_dir, frame_idx = self.frame_index[i]
        reader = self.reader(scene_dir)
        meta = reader.meta
        cam = self._camera
        check_camera_calib(meta, cam)

        frame = reader.frame(frame_idx)
        intr = meta["intrinsics"][cam]
        extrinsic6 = [float(v) for v in meta["extrinsics"][cam]]
        intrinsics4 = [float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"])]

        pose = [float(v) for v in frame["ego"]["transform"]]
        world_vel = np.array(frame["ego"]["velocity"], dtype=np.float64)
        ego_vel = (world_to_ego(pose)[:3, :3] @ world_vel)[:2]        # 世界速度 → ego 系 (前, 右)

        waypoints, valid = self._trajectory(reader, frame_idx, pose)
        sector = dt.sector_of(waypoints, valid, self._fov, self._num_modes)
        target_point = self._target_point(meta, pose)

        depth = np.ascontiguousarray(frame["depth"][cam]).astype(np.float32)
        risk = dt.risk_field(depth, intrinsics4, extrinsic6, self._bev, self._fov, self._depth_max_m)
        drivable = self._hd_map(meta["map"]).drivable_bev(pose, self._bev, self._cfg_data.lane_half_width_m)
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
            "risk": torch.from_numpy(risk),
            "drivable": torch.from_numpy(drivable),
            "distribution": torch.from_numpy(distribution),
            "inview": self._inview,
        }

    def _trajectory(self, reader, frame_idx: int, pose):
        """取同场景未来 num_waypoints 帧 ego 世界位姿，变到当前 ego 系得航点与有效掩码。"""
        last = min(frame_idx + 1 + self._num_waypoints, reader.num_frames)
        future = [reader.frame_meta(j)["ego"]["transform"] for j in range(frame_idx + 1, last)]
        return dt.trajectory_targets(future, pose, self._num_waypoints)

    def _target_point(self, meta, pose):
        """路线终点（世界）变到当前 ego 系，取 xy 作导航目标点。"""
        end = meta["route"]["end"]
        ego_end = transform_points(np.array([end[:3]], dtype=np.float64), world_to_ego(pose))
        return ego_end[0, :2].astype(np.float32)

    def _hd_map(self, map_name: str) -> HdMap:
        """按场景 map 名（去 _Opt 后缀）惰性加载并缓存 HD 地图。"""
        key = map_name.replace("_Opt", "")
        if key not in self._hd_maps:
            path = self._map_dir / self._cfg_data.map_name_template.format(map=key)
            self._hd_maps[key] = HdMap(path)
        return self._hd_maps[key]
