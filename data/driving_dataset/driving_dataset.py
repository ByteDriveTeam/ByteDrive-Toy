"""五帧三目驾驶数据集：独立场景/Agent 占用、检测与轨迹监督。

模块: data/driving_dataset/driving_dataset.py
依赖: numpy, torch, data.single_frame_base, data.driving_targets,
      data.driving_occupancy, data.hd_map, data.lidar_voxelization, vis.data_vis.geometry
读取配置: data.driving.*, data.scene_cache_size, data.dataset.dino_mean/dino_std,
          model.driving.bev/bev_decoder/lidar_fusion/lane_map/traffic_control/trajectory/detection,
          model.physics.depth_max_m
对外接口:
    - DrivingDataset(cfg) -> torch.utils.data.Dataset
说明: 四帧历史图像按真实位姿使用 4×4 刚性变换；场景与 Agent 占用分别生成并按需缓存。
      只返回当前帧 LiDAR，三目 BGR 在训练设备上归一化。
"""
from __future__ import annotations

import warnings
from collections import OrderedDict
from typing import Dict

import numpy as np
import torch

from config.schema import Config
from data import driving_targets as dt
from data.driving_occupancy import DrivingOccupancyCache
from data.driving_dataset.checks.driving_dataset_checks import (
    check_behavior_annotations,
    check_camera_calib,
    check_frame_cadence,
    check_ego_box_annotations,
)
from data.hd_map import HdMap
from data.lidar_voxelization import lidar_xyz_to_voxels
from data.single_frame_base import SingleFrameSceneBase, resolve_repo_path
from vis.data_vis.geometry import transform_matrix, transform_points, world_to_ego


__all__ = ["DrivingDataset"]


class DrivingDataset(SingleFrameSceneBase):
    """以当前帧为索引并读取四帧历史的三目驾驶数据集。"""

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
        self._history_frames = drv_data.history_frames
        self._detect_queries = cfg.model.driving.detection.num_queries
        self._detect_steps = cfg.model.driving.detection.future_steps
        self._detect_dt = cfg.model.driving.detection.future_dt_s
        self._occupancy = DrivingOccupancyCache(cfg)
        self._occupancy.check_sources(scene for scene, _ in self.frame_index)
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
        behavior = drv_data.behavior
        self._behavior_params = dt.BehaviorParams(
            behavior.stationary_speed_mps, behavior.acceleration_threshold_mps2,
            behavior.turn_angle_deg, drv_data.lane_half_width_m,
            behavior.traffic_light_semantic_tag, behavior.traffic_light_match_radius_m,
            behavior.traffic_light_seg_margin_px, behavior.traffic_light_min_pixels)
        self._map_dir = resolve_repo_path(drv_data.map_dir)
        self._inview_np = dt.inview_mask(self._bev, self._fov)
        self._hd_maps: Dict[str, HdMap] = {}
        self._state_cache = OrderedDict()  # 每场景 (ego 位姿 [F,6], 标量速度加速度 [F])
        self._missing_lidar_warned = set()

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        """取五帧三目、当前 LiDAR、独立占用与 Detect/轨迹监督。"""
        scene_dir, frame_idx = self.frame_index[i]
        reader = self.reader(scene_dir)
        meta = reader.meta
        cameras = self._cameras
        check_camera_calib(meta, cameras)
        check_frame_cadence(meta, self._previous_offset, self._detect_dt)
        frame = reader.frame(frame_idx, modalities=("depth", "lidar"))
        check_behavior_annotations(meta, frame, cameras)
        pose = [float(value) for value in frame["ego"]["transform"]]
        intrinsics = [meta["intrinsics"][camera] for camera in cameras]
        intrinsic4 = np.asarray([
            [item["fx"], item["fy"], item["cx"], item["cy"]] for item in intrinsics
        ], dtype=np.float32)
        extrinsics = np.asarray([meta["extrinsics"][camera] for camera in cameras],
                                dtype=np.float32)
        rgb = np.stack([frame["rgb"][camera] for camera in cameras])
        depth = (np.stack([np.asarray(frame["depth"][camera], dtype=np.float32)
                           for camera in cameras])
                 if all(camera in frame["depth"] for camera in cameras) else None)
        history_indices = [max(frame_idx - distance * self._previous_offset, 0)
                           for distance in range(self._history_frames, 0, -1)]
        history_valid = np.asarray([
            frame_idx >= distance * self._previous_offset
            for distance in range(self._history_frames, 0, -1)], dtype=bool)
        history_rgb = np.stack([
            np.stack([reader.rgb(index, camera) for camera in cameras])
            if valid else rgb
            for index, valid in zip(history_indices, history_valid)])
        current_from_world = world_to_ego(pose)
        history_to_current = np.stack([
            current_from_world @ transform_matrix(
                reader.frame_meta(index)["ego"]["transform"])
            if valid else np.eye(4)
            for index, valid in zip(history_indices, history_valid)
        ]).astype(np.float32)
        lidar_points, lidar_ids = self._lidar_target_points(
            frame["lidar"], meta, frame["meta"])
        lidar_stats, lidar_occupied, lidar_valid = self._lidar_voxels(
            scene_dir, frame["lidar"], meta, frame["meta"])
        states = self._scene_states(scene_dir, reader)
        state_idx = states["frame_to_index"].get(int(frame["meta"]["frame_id"]))
        if state_idx is None:
            state_idx = int(np.argmin(np.abs(states["times"] - float(frame["meta"]["sim_time"]))))
        waypoints, traj_valid = self._trajectory(states, state_idx, pose)
        target_point = self._target_point(states["poses"], state_idx, pose, meta)
        world_velocity = np.asarray(frame["ego"]["velocity"], dtype=np.float64)
        ego_velocity = (current_from_world[:2, :2] @ world_velocity[:2]).astype(np.float32)
        hd_map = self._hd_map(meta["map"])
        speed = float(np.linalg.norm(world_velocity[:2]))
        traffic = self._traffic_targets(
            hd_map, states["poses"], state_idx, pose, target_point, meta, frame, speed)
        drivable = hd_map.drivable_bev(
            pose, self._bev, self._cfg_data.lane_half_width_m)
        lane_cfg = self._cfg_data.lane_map
        lane_class, _ = hd_map.lane_map_bev(
            pose, self._bev, lane_cfg.line_width_m,
            lane_cfg.type_to_class, lane_cfg.unknown_class)
        moving = [box for box in frame["bboxes"]
                  if box.get("semantic") in ("vehicle", "pedestrian")]
        visible = self._visible_agents(moving, depth, intrinsics, extrinsics,
                                       pose, lidar_points, lidar_ids)
        scene_occ, scene_mask = self._occupancy.scene(
            scene_dir, frame_idx, pose, intrinsics, extrinsics, rgb.shape[1:3])
        agent_occ, agent_mask = self._occupancy.agent(
            scene_dir, frame_idx, pose, moving, visible, scene_occ,
            intrinsics, extrinsics, rgb.shape[1:3])
        # 缓存按 x 递增索引，BEV 特征行从远到近；只在输出边界翻转以保持缓存几何直观。
        scene_occ, scene_mask, agent_occ, agent_mask = (
            np.ascontiguousarray(np.flip(value, axis=1))
            for value in (scene_occ, scene_mask, agent_occ, agent_mask))
        detect_class, detect_box, detect_future, detect_future_valid = self._detect_targets(
            reader, frame_idx, pose, moving, visible)
        return {
            "rgb": torch.stack([self.bgr_uint8(image) for image in rgb]),
            "history_rgb": torch.stack([
                torch.stack([self.bgr_uint8(image) for image in views])
                for views in history_rgb]),
            "history_to_current": torch.from_numpy(history_to_current),
            "history_valid": torch.from_numpy(history_valid),
            "lidar_stats": lidar_stats,
            "lidar_occupied": lidar_occupied,
            "lidar_valid": torch.tensor(lidar_valid, dtype=torch.bool),
            "intrinsics": torch.from_numpy(intrinsic4),
            "extrinsics": torch.from_numpy(extrinsics),
            "target_point": torch.tensor(target_point, dtype=torch.float32),
            "ego_velocity": torch.from_numpy(ego_velocity),
            "trajectory": torch.from_numpy(waypoints),
            "traj_valid": torch.from_numpy(traj_valid),
            "drivable": torch.from_numpy(drivable.astype(np.float32)),
            "lane_occupancy": torch.from_numpy((lane_class != 0).astype(np.float32)),
            "stop_line": torch.from_numpy(np.asarray(traffic["stop_line"], dtype=np.float32)),
            "scene_occ": torch.from_numpy(scene_occ),
            "scene_occ_mask": torch.from_numpy(scene_mask),
            "agent_occ": torch.from_numpy(agent_occ),
            "agent_occ_mask": torch.from_numpy(agent_mask),
            "detect_class": torch.from_numpy(detect_class),
            "detect_box": torch.from_numpy(detect_box),
            "detect_future": torch.from_numpy(detect_future),
            "detect_future_valid": torch.from_numpy(detect_future_valid),
        }

    def _prepare_occupancy_sample(self, i: int) -> None:
        """只读取当前帧并预生成两份占用缓存，跳过历史图像、地图和轨迹监督。"""
        scene_dir, frame_idx = self.frame_index[i]
        if self._occupancy.contains(scene_dir, frame_idx):
            return
        reader = self.reader(scene_dir)
        meta = reader.meta
        cameras = self._cameras
        check_camera_calib(meta, cameras)
        frame = reader.frame(frame_idx, modalities=("depth", "lidar"))
        pose = [float(value) for value in frame["ego"]["transform"]]
        intrinsics = [meta["intrinsics"][camera] for camera in cameras]
        extrinsics = np.asarray([meta["extrinsics"][camera] for camera in cameras],
                                dtype=np.float32)
        image_shape = frame["rgb"][cameras[0]].shape[:2]
        depth = (np.stack([np.asarray(frame["depth"][camera], dtype=np.float32)
                           for camera in cameras])
                 if all(camera in frame["depth"] for camera in cameras) else None)
        lidar_points, lidar_ids = self._lidar_target_points(
            frame["lidar"], meta, frame["meta"])
        moving = [box for box in frame["bboxes"]
                  if box.get("semantic") in ("vehicle", "pedestrian")]
        visible = self._visible_agents(moving, depth, intrinsics, extrinsics,
                                       pose, lidar_points, lidar_ids)
        scene_occ, _ = self._occupancy.scene(
            scene_dir, frame_idx, pose, intrinsics, extrinsics, image_shape)
        self._occupancy.agent(scene_dir, frame_idx, pose, moving, visible,
                              scene_occ, intrinsics, extrinsics, image_shape)

    def _visible_agents(self, boxes, depth, intrinsics, extrinsics, pose,
                        lidar_points, lidar_ids):
        """复用深度优先、LiDAR 回退的框级可见性判定。"""
        return dt.visible_moving_boxes(
            boxes, depth, intrinsics, pose, extrinsics,
            self._depth_max_m, self._box_min_visible_pixels,
            lidar_points=lidar_points, lidar_object_ids=lidar_ids)

    def _detect_targets(self, reader, frame_idx, pose, boxes, visible):
        """只存储本帧可见运动 Agent；未来轨迹按 actor ID 临近帧匹配。"""
        count = self._detect_queries
        classes = np.full(count, -1, dtype=np.int64)
        box_target = np.zeros((count, 9), dtype=np.float32)
        future = np.zeros((count, self._detect_steps, 2), dtype=np.float32)
        future_valid = np.zeros((count, self._detect_steps), dtype=bool)
        selected = sorted((box for box, keep in zip(boxes, visible) if keep),
                          key=lambda box: np.linalg.norm(
                              np.asarray(box["location"])[:2] - np.asarray(pose)[:2]))[:count]
        next_frames = [reader.frame_meta(min(frame_idx + step, reader.num_frames - 1))
                       if frame_idx + step < reader.num_frames else None
                       for step in range(1, self._detect_steps + 1)]
        current_from_world = world_to_ego(pose)
        for index, box in enumerate(selected):
            classes[index] = 0 if box["semantic"] == "vehicle" else 1
            center = transform_points(np.asarray(box["location"])[None],
                                      current_from_world)[0]
            dimensions = 2 * np.asarray(box["extent"], dtype=np.float32)
            yaw = np.deg2rad(float(box["rotation"][2]) - float(pose[5]))
            matched = [next((item for item in frame["bboxes"]
                             if item.get("id") == box.get("id")), None)
                       if frame is not None else None for frame in next_frames]
            points = [transform_points(np.asarray(item["location"])[None],
                                       current_from_world)[0, :2]
                      if item is not None else None for item in matched]
            for step, point in enumerate(points):
                if point is not None:
                    future[index, step] = point
                    future_valid[index, step] = True
            velocity = ((points[0] - center[:2]) / self._detect_dt
                        if points and points[0] is not None else np.zeros(2))
            box_target[index] = np.r_[center, dimensions, yaw, velocity]
        return classes, box_target, future, future_valid

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
