"""用场景原始框、交通灯、HD 地图及历史运动学判定动作词风险。

模块: data/driving_cache/trajectory_risk.py
依赖: numpy, scipy.spatial, data.hd_map, vis.data_vis.reader, vis.data_vis.geometry
读取配置: data.bevseg.map_dir/map_name_template, data.bevseg.lane_half_width_m,
          model.unified_driving.waypoint_dt_s/risk_jerk_limit_mps3
对外接口:
    - SceneTrajectoryRisk(scene_path, cfg) -> object
    - SceneTrajectoryRisk.gt_trajectory(index, steps) -> np.ndarray
    - evaluate(sample, physical_words) -> (word_risks, gt_risks)
"""

from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from data.hd_map import HdMap
from model.driving_model.unified_driving_model import integrate_planar_accelerations, signed_log
from vis.data_vis.geometry import bbox_corners, transform_points, world_to_ego
from vis.data_vis.reader import SceneReader
from data.carla_data_collector.common.obstacle_geometry import SOLID_SEMANTICS, vertical_overlap


__all__ = ["SceneTrajectoryRisk"]


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _segment_crossings(start, end, line_start, line_end, spatial_index=None):
    """批量判断主车中心线段是否与地图真实线段相交。"""
    if spatial_index is not None:
        tree, half_length = spatial_index
        midpoint = (start + end) * 0.5
        radius = np.linalg.norm(end - start, axis=-1) * 0.5 + half_length
        candidates = tree.query_ball_point(midpoint, radius)
        sizes = np.fromiter(map(len, candidates), dtype=np.int64)
        if not sizes.any():
            return np.zeros(len(start), dtype=bool)
        owner = np.repeat(np.arange(len(start)), sizes)
        indices = np.concatenate(candidates).astype(np.int64)
        direction = end[owner] - start[owner]
        line = line_end[indices] - line_start[indices]
        offset = line_start[indices] - start[owner]
        denominator = _cross(direction, line)
        safe = np.where(np.abs(denominator) < 1e-9, np.inf, denominator)
        along = _cross(offset, line) / safe
        across = _cross(offset, direction) / safe
        crossed = (along >= 0) & (along <= 1) & (across >= 0) & (across <= 1)
        result = np.zeros(len(start), dtype=bool)
        np.logical_or.at(result, owner, crossed)
        return result
    direction = end[:, None] - start[:, None]
    line = line_end[None] - line_start[None]
    offset = line_start[None] - start[:, None]
    denominator = _cross(direction, line)
    safe = np.where(np.abs(denominator) < 1e-9, np.inf, denominator)
    along = _cross(offset, line) / safe
    across = _cross(offset, direction) / safe
    return ((along >= 0) & (along <= 1) & (across >= 0) & (across <= 1)).any(axis=1)


def _box_corners(centers, headings, front, side):
    forward = np.stack((np.cos(headings), np.sin(headings)), axis=-1)
    right = np.stack((-np.sin(headings), np.cos(headings)), axis=-1)
    return centers[:, None] + np.array([1, 1, -1, -1])[None, :, None] * front * forward[:, None] \
        + np.array([1, -1, -1, 1])[None, :, None] * side * right[:, None]


def _overlap_polygons(ego, actor):
    """四个分离轴上都有投影重叠时，两有向车体相交。"""
    edge_ego = ego[:, [1, 2]] - ego[:, [0, 1]]
    edge_actor = actor[[1, 2]] - actor[[0, 1]]
    axes = np.concatenate((edge_ego, np.broadcast_to(edge_actor, edge_ego.shape)), axis=1)
    axes = np.stack((-axes[..., 1], axes[..., 0]), axis=-1)
    ego_projection = np.einsum("nkd,njd->nkj", axes, ego)
    actor_projection = np.einsum("nkd,jd->nkj", axes, actor)
    return ((ego_projection.min(-1) <= actor_projection.max(-1))
            & (actor_projection.min(-1) <= ego_projection.max(-1))).all(-1)


class SceneTrajectoryRisk:
    """同一场景复用轻量帧元数据与矢量地图。"""

    def __init__(self, scene_path, cfg):
        self.cfg = cfg
        self.reader = SceneReader(scene_path, open_videos=False)
        self.meta = self.reader.meta
        self.frames = [self.reader.frame_meta(i) for i in range(self.reader.num_frames)]
        self.frame_times = np.asarray([row["sim_time"] for row in self.frames])
        self.kinematics = self.reader.kinematics()
        self.kin_times = np.asarray([row["sim_time"] for row in self.kinematics])
        self.kin_positions = np.asarray(
            [row["ego"]["transform"][:3] for row in self.kinematics], dtype=np.float64)
        map_root = Path(cfg.data.bevseg.map_dir)
        if not map_root.is_absolute():
            map_root = Path(__file__).resolve().parents[2] / map_root
        name = str(self.meta["map"]).replace("_Opt", "")
        self.map = HdMap(map_root / cfg.data.bevseg.map_name_template.format(map=name))
        types = np.asarray(self.map._line_types)
        self.dividers = self._segments(np.isin(types, ("Broken", "Solid")))
        self.boundaries = self._segments(types == "NONE")
        self.centers = self._segments(types == "Center")
        self.divider_index = self._segment_index(self.dividers)
        self.boundary_index = self._segment_index(self.boundaries)
        center_start, center_end = self.centers
        if len(center_start) == 0:
            raise ValueError("风险监督缺少车道中心线")
        self.center_tree = cKDTree((center_start + center_end) * 0.5)
        self.center_half_length = np.linalg.norm(center_end - center_start, axis=-1).max() * 0.5

    def close(self):
        """释放 LMDB 只读句柄。"""
        self.reader.close()

    def gt_trajectory(self, index, steps):
        """在当前帧坐标系按统一预测时间轴插值完整 GT 笛卡尔航点。"""
        dt = self.cfg.model.unified_driving.waypoint_dt_s
        start = float(self.frames[index]["sim_time"])
        future_times = start + np.arange(1, steps + 1) * dt
        if future_times[-1] > self.kin_times[-1]:
            raise ValueError("GT 轨迹缺少完整未来运动学时间窗口")
        future = np.column_stack([
            np.interp(future_times, self.kin_times, self.kin_positions[:, axis])
            for axis in range(3)])
        pose = self.frames[index]["ego"]["transform"]
        return transform_points(future, world_to_ego(pose))[:, :2].astype(np.float32)

    def _segments(self, selected):
        lines = [poly[:, :2] for poly, keep in zip(self.map._polylines, selected) if keep]
        starts = [line[:-1] for line in lines if len(line) > 1]
        ends = [line[1:] for line in lines if len(line) > 1]
        if not starts:
            return np.empty((0, 2)), np.empty((0, 2))
        return np.concatenate(starts), np.concatenate(ends)

    @staticmethod
    def _segment_index(segments):
        starts, ends = segments
        if len(starts) == 0:
            return None
        return cKDTree((starts + ends) * 0.5), np.linalg.norm(ends - starts, axis=-1).max() * 0.5

    def _aligned_frames(self, index, steps):
        start_time = float(self.frames[index]["sim_time"])
        future = start_time + np.arange(1, steps + 1) * self.cfg.model.unified_driving.waypoint_dt_s
        indices = np.searchsorted(self.frame_times, future)
        if (indices >= len(self.frames)).any():
            raise ValueError("风险监督缺少完整未来帧")
        nearest = np.clip(indices - 1, 0, len(self.frames) - 1)
        choose_left = abs(self.frame_times[nearest] - future) < abs(self.frame_times[indices] - future)
        indices = np.where(choose_left, nearest, indices)
        if np.max(abs(self.frame_times[indices] - future)) > self.cfg.model.unified_driving.waypoint_dt_s * 0.51:
            raise ValueError("风险监督未来帧时间戳不对齐")
        return [self.frames[i] for i in indices]

    def _history_control(self, index):
        now = self.frame_times[index]
        end = int(np.searchsorted(self.kin_times, now, side="right"))
        if end < 1:
            raise ValueError("风险监督缺少当前运动学点")
        row = self.kinematics[end - 1]["ego"]
        yaw = np.deg2rad(self.frames[index]["ego"]["transform"][5])
        acceleration = np.asarray(row["acceleration"][:2], dtype=np.float64)
        return np.array([np.cos(yaw) * acceleration[0] + np.sin(yaw) * acceleration[1],
                         -np.sin(yaw) * acceleration[0] + np.cos(yaw) * acceleration[1]])

    def _offroad(self, corners, lane_half_width):
        starts, ends = self.centers
        if len(starts) == 0:
            raise ValueError("风险监督缺少附近的车道中心线")
        points = corners.reshape(-1, 2)
        nearest, _ = self.center_tree.query(points)
        half = self.center_half_length
        offroad = nearest > lane_half_width + half
        uncertain = (nearest >= lane_half_width - half) & ~offroad
        if uncertain.any():
            ambiguous = points[uncertain]
            candidates = self.center_tree.query_ball_point(ambiguous, lane_half_width + half)
            lengths = np.fromiter(map(len, candidates), dtype=np.int64)
            owner = np.repeat(np.arange(len(ambiguous)), lengths)
            indices = np.concatenate(candidates).astype(np.int64)
            segment = ends[indices] - starts[indices]
            relative = ambiguous[owner] - starts[indices]
            along = np.clip((relative * segment).sum(-1)
                            / np.maximum((segment * segment).sum(-1), 1e-12), 0, 1)
            distance = np.linalg.norm(relative - along[:, None] * segment, axis=-1)
            minimum = np.full(len(ambiguous), np.inf)
            np.minimum.at(minimum, owner, distance)
            offroad[uncertain] = minimum > lane_half_width
        return offroad.reshape(corners.shape[:2]).any(-1)

    def _risk_flags(self, local, headings, actions, frame, future, history):
        count, steps = local.shape[:2]
        pose = np.asarray(frame["ego"]["transform"], dtype=np.float64)
        theta = np.deg2rad(pose[5])
        rotation = np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]])
        world = local @ rotation.T + pose[:2]
        world_headings = headings + theta
        ego_boxes = [box for box in frame.get("bboxes", []) if box.get("semantic") == "ego"]
        if not ego_boxes:
            raise ValueError("风险监督缺少自车包围框")
        front, side = np.asarray(ego_boxes[0]["extent"][:2], dtype=np.float64)
        flags = np.zeros((count, 5), dtype=bool)
        previous = np.broadcast_to(pose[:2], (count, 2)).copy()
        previous_front = previous + front * np.array([np.cos(theta), np.sin(theta)])
        for step, metadata in enumerate(future):
            position = world[:, step]
            angle = world_headings[:, step]
            corners = _box_corners(position, angle, front, side)
            future_ego = next((box for box in metadata.get('bboxes', [])
                               if box.get('semantic') == 'ego'), ego_boxes[0])
            actors = list(metadata.get('bboxes', [])) + self._near_static_boxes(corners, future_ego)
            for actor in actors:
                if actor.get("semantic") not in SOLID_SEMANTICS or not vertical_overlap(actor, future_ego):
                    continue
                actor_corners = bbox_corners(actor)[:4, :2]
                flags[:, 0] |= _overlap_polygons(corners, actor_corners)
            front_position = position + front * np.stack((np.cos(angle), np.sin(angle)), axis=-1)
            red_ids = {light["id"] for light in metadata.get("traffic_light_states", [])
                       if light.get("state") == "red"}
            for light in self.meta.get("traffic_lights", []):
                if light["id"] not in red_ids:
                    continue
                for stop in light.get("stop_waypoints", []):
                    stop_pose = stop.get("transform", [])
                    if len(stop_pose) < 6:
                        continue
                    yaw = np.deg2rad(stop_pose[5])
                    lateral = np.array([-np.sin(yaw), np.cos(yaw)])
                    half = float(stop.get("lane_width", 0)) * 0.5 + side
                    center = np.asarray(stop_pose[:2])
                    flags[:, 1] |= _segment_crossings(
                        previous_front, front_position,
                        (center - lateral * half)[None], (center + lateral * half)[None])
            if self.divider_index is not None:
                flags[:, 3] |= _segment_crossings(
                    previous, position, *self.dividers, self.divider_index)
            if self.boundary_index is not None:
                flags[:, 4] |= _segment_crossings(
                    previous, position, *self.boundaries, self.boundary_index)
            flags[:, 4] |= self._offroad(corners, self.cfg.data.bevseg.lane_half_width_m)
            previous, previous_front = position, front_position
        dt = self.cfg.model.unified_driving.waypoint_dt_s
        flags[:, 2] = (np.linalg.norm(actions[:, 0] - history, axis=-1) / dt
                       > self.cfg.model.unified_driving.risk_jerk_limit_mps3)
        return flags

    def _near_static_boxes(self, corners, ego):
        """Reuse scene-level solids and reject distant/other-level boxes in bulk."""
        if not hasattr(self, '_solid_static_boxes'):
            self._solid_static_boxes = [box for box in self.meta.get('static_bboxes', [])
                                        if box.get('semantic') in ('static', 'pole')]
            self._solid_static_bounds = np.asarray([
                [*bbox_corners(box)[:, :2].min(0), *bbox_corners(box)[:, :2].max(0),
                 box['location'][2] - box['extent'][2], box['location'][2] + box['extent'][2]]
                for box in self._solid_static_boxes], dtype=float).reshape(-1, 6)
        bounds = self._solid_static_bounds
        if not len(bounds):
            return []
        lower, upper = corners.min((0, 1)), corners.max((0, 1))
        selected = ((bounds[:, 0] <= upper[0]) & (bounds[:, 1] <= upper[1]) &
                    (bounds[:, 2] >= lower[0]) & (bounds[:, 3] >= lower[1]) &
                    (bounds[:, 4] <= ego['location'][2] + ego['extent'][2]) &
                    (bounds[:, 5] >= ego['location'][2] - ego['extent'][2]))
        return [box for box, keep in zip(self._solid_static_boxes, selected) if keep]

    def evaluate(self, sample, physical_words):
        """输出 [词数,5] 和 [5] 风险；缺失真值则明确拒绝该切片。"""
        index = int(sample["index"])
        words = np.asarray(physical_words, dtype=np.float32)
        future = self._aligned_frames(index, words.shape[1])
        history = self._history_control(index)
        raw_state = torch.as_tensor(sample["ego_state"]).float()
        signed_state = signed_log(raw_state).expand(len(words), -1)
        integrated = integrate_planar_accelerations(
            torch.from_numpy(words), signed_state,
            dt_s=self.cfg.model.unified_driving.waypoint_dt_s).numpy()
        local = integrated[..., :2]
        headings = integrated[..., 2]
        word_flags = self._risk_flags(local, headings, words,
                                      self.frames[index], future, history)
        gt_local = np.asarray(sample["trajectory"], dtype=np.float32)
        current_yaw = np.deg2rad(self.frames[index]["ego"]["transform"][5])
        gt_heading = np.unwrap(np.deg2rad(
            [row["ego"]["transform"][5] for row in future])) - current_yaw
        gt_actions = np.asarray(sample.get("trajectory_actions"), dtype=np.float32)
        if gt_actions.shape != (words.shape[1], 2):
            raise ValueError("风险监督需要物理单位的 GT 动作")
        gt_flags = self._risk_flags(
            gt_local[None], gt_heading[None], gt_actions[None],
            self.frames[index], future, history)[0]
        return torch.from_numpy(word_flags), torch.from_numpy(gt_flags)
