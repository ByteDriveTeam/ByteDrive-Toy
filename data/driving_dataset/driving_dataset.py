"""驾驶模型双帧三目+LiDAR 数据集：产双帧输入、体素统计、帧间变换与驾驶多任务监督。

模块: data/driving_dataset/driving_dataset.py
依赖: torch, numpy, warnings, config.schema.Config, data.single_frame_base.SingleFrameSceneBase,
      data.lidar_voxelization,
      data.driving_targets, data.hd_map.HdMap, vis.data_vis.geometry, data.driving_dataset.checks.*
读取配置:
    data.driving.scene_root / cameras / map_dir / map_name_template / previous_frame_offset /
        dist_sigma_m / lane_half_width_m
    data.scene_cache_size
    data.driving.lane_map.line_width_m / centerline_match_radius_m / type_to_class / unknown_class
    data.driving.traffic_control.route_lookahead_m / route_corridor_m / line_expand_m /
        actor_match_radius_m / stop_margin_m / reaction_time_s / comfortable_decel_mps2
    data.driving.box_min_visible_pixels
    data.driving.target_min_m / target_max_m（目标点采样距离窗口）
    data.driving.behavior.stationary_speed_mps / acceleration_threshold_mps2 / turn_angle_deg /
        traffic_light_semantic_tag / traffic_light_match_radius_m / traffic_light_seg_margin_px /
        traffic_light_min_pixels
    data.dataset.dino_mean / dino_std
    model.driving.bev.x_min_m / x_max_m / y_min_m / y_max_m / fov_deg
    model.driving.lidar_fusion.voxel_size_m
    model.driving.bev_decoder.up_channels（推导场分辨率 = bev.height/width · 2^L）
    model.driving.lane_map.class_names（定位中心线类别索引）
    model.driving.trajectory.num_waypoints / waypoint_dt_s
    model.driving.traffic_control.state_names
    model.physics.depth_max_m（风险场包络排除超范围/天空像素）
对外接口:
    - DrivingDataset(cfg) -> torch.utils.data.Dataset
        __getitem__(i) -> dict[str, Tensor]
说明: 复用 SingleFrameSceneBase 的索引/reader 缓存；RGB 以 BGR uint8 紧凑返回并在设备侧归一化。
      三路相机严格按 data.driving.cameras 堆叠；
      当前/历史语义 LiDAR 先按各帧真实自车有向 Box 剔除车体内点，再在 CPU 上编码为 0.5m 体素中心
      相对 XYZ 米制均值与总体标准差；旧场景缺失时按场景告警并旁路。
      每个样本同时返回同场景上一帧三目 RGB 及把
      上一帧 ego 平面坐标变到当前 ego 系的 3×3 刚性矩阵；场景开头返回当前 RGB、identity 与 previous_valid=0。
      轨迹 GT 优先从独立运动学时间轴按 10Hz 取未来 num_waypoints 个 ego 世界位姿，再经 world_to_ego
      变到当前 ego 系；旧场景自动回退低频逐帧状态。行为 GT 为固定八类
      多热向量，组合当前速度/帧间加速度、未来轨迹、动态 Agent 框与路线相关交通灯状态；红灯停车在接近阶段即激活。
      新场景逐帧携带 CARLA 原生受控车道/Agent 规划关联结果，直接生成停止线；旧场景缺该字段时自动回退到
      HD Map 触发区与未来专家路线走廊相交算法，无需迁移历史 LMDB。
      目标点沿未来自车轨迹搜距当前 target_min~target_max m 的点随机取一（近端引导 + 鲁棒），变到 ego 系；
      当前世界速度同步旋转到 ego 平面，二者共同作为规划条件。
      风险场优先由 GT 深度反投影包络、缺失时回退 LiDAR；可行驶场先由 HD 地图按位姿栅格化，再扣除由
      深度或 LiDAR 确认可见的
      vehicle/pedestrian box 占用（运动类别间不分类，ego/静态环境框排除），并转成道路外/占用距离场供轨迹约束使用；
      道路线图由 HD Map 的 Type 与每点 yaw 栅格化为类别和有向单位切向量；GT 可靠贴近的中心线折线
      另生成米制距离场，规控主动偏离超过配置阈值的航点不参与贴线监督；分布场由 GT 航点高斯软化，视场掩码为常量
      （构造期预算）。全帧 ego 位姿与速度加速度采用同一有界 LRU 场景缓存，供轨迹/行为/目标点复用且不随场景数涨内存。场分辨率与
      模型上采样输出一致（Hb·2^L）。HD 地图按场景 map 名（去 _Opt 后缀）惰性加载并缓存。几何投影复用
      vis.data_vis.geometry / data.driving_targets。
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from typing import Dict

import numpy as np
import torch

from config.schema import Config
from data import driving_targets as dt
from data.driving_dataset.checks.driving_dataset_checks import (
    check_behavior_annotations,
    check_camera_calib,
    check_ego_box_annotations,
)
from data.hd_map import HdMap, offroad_distance_field
from data.lidar_voxelization import lidar_xyz_to_voxels
from data.single_frame_base import SingleFrameSceneBase, resolve_repo_path
from vis.data_vis.geometry import transform_matrix, transform_points, world_to_ego


__all__ = ["DrivingDataset"]


class DrivingDataset(SingleFrameSceneBase):
    """以当前帧为索引、同时读取上一帧的双帧三目驾驶数据集。"""

    def __init__(self, cfg: Config) -> None:
        drv_data = cfg.data.driving
        super().__init__(drv_data.scene_root, drv_data.cameras[0],
                         cfg.data.dataset.dino_mean, cfg.data.dataset.dino_std,
                         cfg.data.scene_cache_size)
        self._cfg_data = drv_data
        self._cameras = tuple(drv_data.cameras)
        bev = cfg.model.driving.bev
        self._bev_geometry = bev
        self._lidar_voxel_size = cfg.model.driving.lidar_fusion.voxel_size_m
        self._fov = bev.fov_deg
        self._previous_offset = drv_data.previous_frame_offset
        # 场分辨率 = BEV 工作分辨率 · 统一解码头上采样倍率
        scale = 2 ** len(cfg.model.driving.bev_decoder.up_channels)
        self._bev = dt.BevParams(bev.x_min_m, bev.x_max_m, bev.y_min_m, bev.y_max_m,
                                 bev.height * scale, bev.width * scale)
        self._num_waypoints = cfg.model.driving.trajectory.num_waypoints
        self._waypoint_dt = cfg.model.driving.trajectory.waypoint_dt_s
        self._depth_max_m = cfg.model.physics.depth_max_m  # 风险场包络排除超范围/天空像素
        self._box_min_visible_pixels = drv_data.box_min_visible_pixels
        self._target_min = drv_data.target_min_m
        self._target_max = drv_data.target_max_m
        self._traffic_cfg = drv_data.traffic_control
        self._traffic_state_names = cfg.model.driving.traffic_control.state_names
        centerline_class = cfg.model.driving.lane_map.class_names.index("centerline")
        self._centerline_types = tuple(
            name for name, class_id in drv_data.lane_map.type_to_class.items()
            if class_id == centerline_class)
        behavior = drv_data.behavior
        self._behavior_params = dt.BehaviorParams(
            behavior.stationary_speed_mps, behavior.acceleration_threshold_mps2,
            behavior.turn_angle_deg, drv_data.lane_half_width_m,
            behavior.traffic_light_semantic_tag, behavior.traffic_light_match_radius_m,
            behavior.traffic_light_seg_margin_px, behavior.traffic_light_min_pixels)
        self._map_dir = resolve_repo_path(drv_data.map_dir)
        self._inview_np = dt.inview_mask(self._bev, self._fov)
        self._inview = torch.from_numpy(self._inview_np).to(torch.uint8)  # 紧凑常量，预算一次
        self._hd_maps: Dict[str, HdMap] = {}
        self._state_cache = OrderedDict()  # 每场景 (ego 位姿 [F,6], 标量速度加速度 [F])
        self._missing_lidar_warned = set()

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        scene_dir, frame_idx = self.frame_index[i]
        reader = self.reader(scene_dir)
        meta = reader.meta
        cameras = self._cameras
        check_camera_calib(meta, cameras)

        previous_idx = max(frame_idx - self._previous_offset, 0)
        previous_valid = float(frame_idx >= self._previous_offset)
        previous_meta = reader.frame_meta(previous_idx) if previous_valid else None
        previous_rgb = (
            np.stack([reader.rgb(previous_idx, camera) for camera in cameras])
            if previous_valid else None)

        frame = reader.frame(frame_idx, modalities=("depth", "semantic", "lidar"))
        check_behavior_annotations(meta, frame, cameras)
        intrinsics = [meta["intrinsics"][camera] for camera in cameras]
        extrinsics = np.asarray(
            [meta["extrinsics"][camera] for camera in cameras], dtype=np.float32)
        intrinsics4 = np.asarray([
            [intr["fx"], intr["fy"], intr["cx"], intr["cy"]] for intr in intrinsics
        ], dtype=np.float32)
        rgb = np.stack([frame["rgb"][camera] for camera in cameras])
        depth = (
            np.stack([
                np.ascontiguousarray(frame["depth"][camera]).astype(np.float32)
                for camera in cameras
            ])
            if all(camera in frame["depth"] for camera in cameras) else None
        )
        semantic = (
            np.stack([
                np.ascontiguousarray(frame["semantic"][camera]) for camera in cameras
            ])
            if all(camera in frame["semantic"] for camera in cameras) else None
        )

        pose = [float(v) for v in frame["ego"]["transform"]]
        world_vel = np.array(frame["ego"]["velocity"], dtype=np.float64)
        previous_meta = previous_meta or frame["meta"]
        previous_rgb = previous_rgb if previous_rgb is not None else rgb
        previous_lidar = reader.lidar(previous_idx) if previous_valid else frame["lidar"]
        lidar_points, lidar_object_ids = self._lidar_target_points(
            frame["lidar"], meta, frame["meta"])
        lidar_stats, lidar_occupied, lidar_valid = self._lidar_voxels(
            scene_dir, frame["lidar"], meta, frame["meta"])
        previous_lidar_stats, previous_lidar_occupied, previous_lidar_valid = \
            self._lidar_voxels(scene_dir, previous_lidar, meta, previous_meta)
        previous_pose = [float(v) for v in previous_meta["ego"]["transform"]]
        previous_to_current = _planar_previous_to_current(previous_pose, pose)

        states = self._scene_states(scene_dir, reader)
        state_idx = states["frame_to_index"].get(int(frame["meta"]["frame_id"]))
        if state_idx is None:
            state_idx = int(np.argmin(np.abs(states["times"] - float(frame["meta"]["sim_time"]))))
        poses, accelerations = states["poses"], states["accelerations"]
        waypoints, valid = self._trajectory(states, state_idx, pose)
        ego_extent = np.asarray(self._ego_box(frame["meta"])["extent"][:2], dtype=np.float32)
        target_point = self._target_point(poses, state_idx, pose, meta)
        ego_velocity = (world_to_ego(pose)[:2, :2] @ world_vel[:2]).astype(np.float32)
        hd_map = self._hd_map(meta["map"])
        speed_mps = float(np.linalg.norm(world_vel[:2]))
        traffic = self._traffic_targets(
            hd_map, poses, state_idx, pose, target_point, meta, frame, speed_mps)
        behavior = dt.behavior_targets(
            waypoints, valid, speed_mps, float(accelerations[state_idx]),
            frame["bboxes"], meta["traffic_lights"], frame["traffic_light_states"],
            meta["static_bboxes"], semantic, pose, intrinsics, extrinsics,
            self._bev, self._fov, self._behavior_params,
            red_light_relevant=bool(traffic["red_stop_valid"]))

        risk = dt.risk_field(
            depth, intrinsics4, extrinsics, self._bev, self._fov, self._depth_max_m,
            lidar_points=lidar_points)
        map_drivable = hd_map.drivable_bev(
            pose, self._bev, self._cfg_data.lane_half_width_m)
        lane_cfg = self._cfg_data.lane_map
        lane_class, lane_direction = hd_map.lane_map_bev(
            pose, self._bev, lane_cfg.line_width_m,
            lane_cfg.type_to_class, lane_cfg.unknown_class)
        gt_centerline_distance, gt_centerline_valid = hd_map.gt_centerline_distance_bev(
            pose, waypoints, valid, self._bev, self._centerline_types,
            lane_cfg.centerline_match_radius_m)
        box_occupancy = dt.visible_moving_box_occupancy(
            frame["bboxes"], depth, intrinsics, pose, extrinsics,
            self._bev, self._depth_max_m, self._box_min_visible_pixels,
            lidar_points=lidar_points, lidar_object_ids=lidar_object_ids)
        drivable = map_drivable * (1.0 - box_occupancy)
        offroad_distance = offroad_distance_field(drivable, self._bev)
        distribution = dt.distribution_field(waypoints, valid, self._bev, self._cfg_data.dist_sigma_m)

        sample = {
            "rgb": torch.stack([self.bgr_uint8(image) for image in rgb]),
            "previous_rgb": torch.stack([self.bgr_uint8(image) for image in previous_rgb]),
            "previous_to_current": torch.from_numpy(previous_to_current),
            "previous_valid": torch.tensor(previous_valid, dtype=torch.float32),
            "lidar_stats": lidar_stats,
            "lidar_occupied": lidar_occupied,
            "lidar_valid": torch.tensor(lidar_valid, dtype=torch.float32),
            "previous_lidar_stats": previous_lidar_stats,
            "previous_lidar_occupied": previous_lidar_occupied,
            "previous_lidar_valid": torch.tensor(
                previous_lidar_valid, dtype=torch.float32),
            "intrinsics": torch.from_numpy(intrinsics4),
            "extrinsics": torch.from_numpy(extrinsics),
            "target_point": torch.tensor(target_point, dtype=torch.float32),
            "ego_velocity": torch.from_numpy(ego_velocity),
            "ego_extent": torch.from_numpy(ego_extent),
            "trajectory": torch.from_numpy(waypoints),
            "traj_valid": torch.from_numpy(valid),
            "behavior": torch.from_numpy(behavior),
            "risk": torch.from_numpy(risk),
            "drivable": torch.from_numpy(drivable),
            "lane_class": torch.from_numpy(lane_class.astype(np.uint8)),
            "lane_direction": torch.from_numpy(lane_direction),
            "gt_centerline_distance": torch.from_numpy(gt_centerline_distance),
            "gt_centerline_valid": torch.from_numpy(gt_centerline_valid),
            "offroad_distance": torch.from_numpy(offroad_distance),
            "distribution": torch.from_numpy(distribution),
            "inview": self._inview,
        }
        compact_uint8 = {"stop_line", "traffic_light_state", "traffic_light_state_valid"}
        sample.update({
            name: (
                torch.from_numpy(value).to(torch.uint8)
                if isinstance(value, np.ndarray) and name in compact_uint8
                else torch.from_numpy(value)
                if isinstance(value, np.ndarray)
                else torch.tensor(value, dtype=torch.float32)
            )
            for name, value in traffic.items()
        })
        return sample

    def _lidar_voxels(self, scene_dir, lidar, meta, frame_meta):
        """把结构化语义 LiDAR 剔除自车 Box 后转为 ego 系体素统计；缺失场景严格标为无效。"""
        extrinsic = meta.get("lidar_extrinsic")
        if lidar is None or extrinsic is None:
            key = str(scene_dir)
            if key not in self._missing_lidar_warned:
                warnings.warn("场景 {} 缺失 LiDAR，驾驶模型将严格旁路该模态。".format(
                    scene_dir.name), RuntimeWarning)
                self._missing_lidar_warned.add(key)
            stats, occupied = lidar_xyz_to_voxels(
                np.empty((0, 3), dtype=np.float32), (0.0, 0.0, 0.0),
                self._ego_box(frame_meta), self._bev_geometry,
                self._lidar_voxel_size)
            return stats, occupied, 0.0
        xyz = np.stack((lidar["x"], lidar["y"], lidar["z"]), axis=1)
        stats, occupied = lidar_xyz_to_voxels(
            xyz, extrinsic, self._ego_box(frame_meta),
            self._bev_geometry, self._lidar_voxel_size)
        return stats, occupied, 1.0

    def _lidar_target_points(self, lidar, meta, frame_meta):
        """把原始 LiDAR 转到 ego 系并剔除自车点，供监督目标在缺深度时回退。"""
        extrinsic = meta.get("lidar_extrinsic")
        if lidar is None or extrinsic is None:
            return None, None
        points = np.stack((lidar["x"], lidar["y"], lidar["z"]), axis=1).astype(
            np.float64, copy=False)
        points = points + np.asarray(extrinsic, dtype=np.float64)
        ego_box = self._ego_box(frame_meta)
        box_transform = np.asarray(ego_box["transform"], dtype=np.float64)
        box_local = (points - box_transform[:3, 3]) @ box_transform[:3, :3]
        keep = np.any(np.abs(box_local) > np.asarray(ego_box["extent"]), axis=1)
        object_ids = (
            np.asarray(lidar["obj_idx"])[keep]
            if lidar.dtype.names is not None and "obj_idx" in lidar.dtype.names else None
        )
        return points[keep], object_ids

    @staticmethod
    def _ego_box(frame_meta):
        """把逐帧世界系 ego Box 转成主车局部系有向 Box。"""
        ego_boxes = [
            box for box in frame_meta["bboxes"] if box.get("semantic") == "ego"
        ]
        check_ego_box_annotations(ego_boxes)
        box = ego_boxes[0]
        box_pose = box["location"] + box["rotation"]
        ego_pose = frame_meta["ego"]["transform"]
        return {
            "transform": world_to_ego(ego_pose) @ transform_matrix(box_pose),
            "extent": box["extent"],
        }

    def _scene_states(self, scene_dir, reader):
        """以有界 LRU 缓存异频运动学；旧场景由低频逐帧状态自动回退。"""
        key = str(scene_dir)
        state = self._state_cache.pop(key, None)
        if state is None:
            samples = reader.kinematics()
            poses = np.array([sample["ego"]["transform"] for sample in samples], dtype=np.float64)
            velocities = np.array([sample["ego"]["velocity"] for sample in samples], dtype=np.float64)
            sim_times = np.array([sample["sim_time"] for sample in samples], dtype=np.float64)
            state = {
                "poses": poses,
                "times": sim_times,
                "accelerations": dt.speed_accelerations(velocities, sim_times),
                "frame_to_index": {
                    int(sample["frame_id"]): index for index, sample in enumerate(samples)},
            }
        self._state_cache[key] = state
        if len(self._state_cache) > self._scene_cache_size:
            self._state_cache.popitem(last=False)
        return state

    def _trajectory(self, states, state_idx: int, pose):
        """按配置点间隔插值运动学位姿，生成固定 10Hz 航点监督。"""
        poses, times = states["poses"], states["times"]
        target_times = times[state_idx] + self._waypoint_dt * np.arange(
            1, self._num_waypoints + 1, dtype=np.float64)
        valid = target_times <= times[-1] + np.finfo(np.float64).eps * max(abs(times[-1]), 1.0)
        valid_times = target_times[valid]
        future = list(np.column_stack([
            np.interp(valid_times, times, poses[:, column])
            for column in range(poses.shape[1])
        ]))
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
        """按旧版交通控制前视距离截取未来专家路径；长时间等灯时仍能延伸到路口之后。"""
        future = poses[frame_idx + 1:, :3]
        future_ego = transform_points(future, world_to_ego(pose))[:, :2].astype(np.float32)
        route = np.vstack((np.zeros((1, 2), dtype=np.float32), future_ego))
        if len(route) < 2 or np.linalg.norm(target_point) > np.linalg.norm(route[-1]) + 1e-3:
            route = np.vstack((route, target_point))
        arclength = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]
        lookahead = self._traffic_cfg.route_lookahead_m
        end = int(np.searchsorted(arclength, lookahead, side="left"))
        if end < len(route):
            start = end - 1
            ratio = (lookahead - arclength[start]) / (arclength[end] - arclength[start])
            boundary = route[start] + ratio * (route[end] - route[start])
            route = np.vstack((route[:end], boundary.astype(np.float32)))
        return route

    def _traffic_targets(self, hd_map, poses, frame_idx, pose, target_point, meta, frame, speed_mps):
        """生成路线相关交通控制监督，并用视场与舒适制动距离门控红灯停车约束。"""
        traffic = hd_map.traffic_control_bev(
            pose, self._route_polyline(poses, frame_idx, pose, target_point),
            meta["traffic_lights"], frame["traffic_light_states"], self._bev,
            self._traffic_cfg.route_corridor_m, self._traffic_cfg.line_expand_m,
            self._traffic_cfg.actor_match_radius_m, self._traffic_state_names,
            frame["meta"].get("relevant_traffic_control"),
            annotation_version=self._traffic_cfg.annotation_version)
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
