"""HD 地图：加载车道折线与交通灯触发区，生成道路、停止线及越界监督。

模块: data/hd_map/hd_map.py
依赖: numpy, cv2, math, vis.data_vis.geometry(world_to_ego/transform_points),
      data.driving_targets(ego_xy_to_pixel/BevParams), data.hd_map.checks.hd_map_checks
读取配置: —（地图路径、BEV 几何、车道半宽由调用方传入，来源 config.data.driving 与 config.model.driving.bev）
对外接口:
    - HdMap(npz_path) -> HdMap
        .drivable_bev(ego_pose6, bev, lane_half_width_m) -> (H,W) float32   # 1=地图可行驶
        .lane_map_bev(ego_pose6, bev, line_width_m, type_to_class, unknown_class)
            -> (class_map[H,W], direction[2,H,W])
        .traffic_control_bev(ego_pose6, route_xy, traffic_lights, states, bev, ...)
            -> dict[str, ndarray]                                           # 相关停止线/灯色/越线几何
    - offroad_distance_field(drivable, bev) -> (H,W) float32                # 可行驶处=0，越界距离（米）
说明: npz 结构为 `arr[(road_id, {lane_id: [ {Points:[((x,y,z),(rpy),...)...], Type, Color, ...}, ... ]})]`；
      解析世界系 xyz、Type 与每点 yaw 对应的有向单位切向量。栅格化时按 ego 世界位姿把折线与方向搬到 ego 系
      （复用 vis.data_vis.geometry 的 CARLA
      变换，避免重复实现），投到 BEV 像素后用 cv2 粗线（宽 = 2·车道半宽）画出，其并集近似路面可行驶区域——
      稠密且不受语义投影稀疏之苦（用户口径）。交通灯相关性由「触发区与当前路线走廊相交」确定；若多条相交，取沿
      路线弧长最先到达者，避免横向道路或转弯后的灯误关联。逐折线用外接圆半径对 ego 做粗过滤，仅栅格化附近折线。
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from data.driving_targets import BevParams, ego_xy_to_pixel
from data.hd_map.checks.hd_map_checks import (
    check_drivable_mask,
    check_map_path,
    check_polylines,
    check_traffic_control_inputs,
)
from vis.data_vis.geometry import transform_points, world_to_ego


__all__ = ["HdMap", "offroad_distance_field"]


class HdMap:
    """车道折线地图 → ego BEV 可行驶掩码与类别/方向道路线图栅格化器。"""

    def __init__(self, npz_path) -> None:
        check_map_path(npz_path)
        self._path = str(npz_path)
        arr = np.load(self._path, allow_pickle=True)["arr"]
        lanes = _parse_lanes(arr)
        self._polylines = [lane[0] for lane in lanes]
        self._directions = [lane[1] for lane in lanes]
        self._line_types = [lane[2] for lane in lanes]
        check_polylines(self._polylines, self._path)
        # 逐折线外接圆（世界系）：中心与半径，供按 ego 距离粗过滤
        self._centers = np.array([p[:, :2].mean(0) for p in self._polylines], dtype=np.float64)
        self._radii = np.array(
            [np.linalg.norm(p[:, :2] - c, axis=1).max() for p, c in zip(self._polylines, self._centers)],
            dtype=np.float64)
        controls = _parse_traffic_controls(arr)
        self._control_polygons = [item[0] for item in controls]
        self._control_parent_xy = np.array([item[1] for item in controls], dtype=np.float64)

    def drivable_bev(self, ego_pose6, bev: BevParams, lane_half_width_m: float) -> np.ndarray:
        """按 ego 世界位姿栅格化 BEV 可行驶掩码 `(H, W)`（1=可行驶）。"""
        mask = np.zeros((bev.height, bev.width), dtype=np.uint8)
        ego_xy = np.array(ego_pose6[:2], dtype=np.float64)
        reach = math.hypot(bev.x_max - bev.x_min, bev.y_max - bev.y_min)  # BEV 对角覆盖半径
        near = np.nonzero(np.linalg.norm(self._centers - ego_xy, axis=1) - self._radii < reach)[0]
        if near.size == 0:
            return mask.astype(np.float32)

        w2e = world_to_ego(ego_pose6)
        px_per_m = bev.width / (bev.y_max - bev.y_min)               # 方形 BEV：x/y 每米像素数一致
        thickness = max(int(round(2.0 * lane_half_width_m * px_per_m)), 1)
        for i in near:
            ego_xyz = transform_points(self._polylines[i], w2e)     # 世界→ego
            rows, cols = ego_xy_to_pixel(ego_xyz[:, :2], bev)
            pts = np.stack((cols, rows), axis=1).round().astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(mask, [pts], isClosed=False, color=1, thickness=thickness)
        return mask.astype(np.float32)

    def lane_map_bev(self, ego_pose6, bev: BevParams, line_width_m: float,
                     type_to_class, unknown_class: int):
        """按 ego 位姿生成道路线类别 `(H,W)` 与有向单位切向量 `(2,H,W)`。"""
        class_map = np.zeros((bev.height, bev.width), dtype=np.int64)
        direction = np.zeros((2, bev.height, bev.width), dtype=np.float32)
        ego_xy = np.asarray(ego_pose6[:2], dtype=np.float64)
        reach = math.hypot(bev.x_max - bev.x_min, bev.y_max - bev.y_min)
        near = np.nonzero(np.linalg.norm(self._centers - ego_xy, axis=1) - self._radii < reach)[0]
        if near.size == 0:
            return class_map, direction

        w2e = world_to_ego(ego_pose6)
        px_per_m = bev.width / (bev.y_max - bev.y_min)
        thickness = max(int(round(line_width_m * px_per_m)), 1)
        class_ids = np.array(
            [type_to_class.get(self._line_types[i], unknown_class) for i in near], dtype=np.int64)
        for index in near[np.argsort(class_ids)]:
            ego_xyz = transform_points(self._polylines[index], w2e)
            ego_direction = self._directions[index] @ w2e[:2, :2].T
            rows, cols = ego_xy_to_pixel(ego_xyz[:, :2], bev)
            _rasterize_lane(
                class_map, direction, rows, cols, ego_direction,
                type_to_class.get(self._line_types[index], unknown_class), thickness)
        return class_map, direction

    def traffic_control_bev(self, ego_pose6, route_xy, traffic_lights, states,
                            bev: BevParams, route_corridor_m: float, line_expand_m: float,
                            actor_match_radius_m: float, state_names):
        """生成与当前路线相关的交通灯停止区及状态监督。

        相关性要求触发区位于 ego 前方、落入 BEV，且与路线走廊相交；多候选取沿路线投影弧长最小者。
        返回的停止点是原始触发区中心，方向取路线在该处的切向，供红灯越线损失定义线前/线后。
        """
        check_traffic_control_inputs(route_xy, state_names)
        out = _empty_traffic_control(bev)
        route = _clean_route(route_xy)
        if len(route) < 2 or not self._control_polygons:
            return out

        route_mask = _route_mask(route, bev, route_corridor_m)
        actor_ids = _match_control_actors(
            self._control_parent_xy, traffic_lights, actor_match_radius_m)
        state_by_id = {item["id"]: item.get("state", "unknown") for item in states}
        selected = _select_relevant_control(
            self._control_polygons, actor_ids, world_to_ego(ego_pose6), route, route_mask, bev)
        if selected is None:
            return out

        arclength, polygon, direction, actor_id = selected
        line = _expanded_polygon_mask(polygon, bev, line_expand_m).astype(np.float32)
        state = state_by_id.get(actor_id, "unknown")
        out["stop_line"] = line
        out["stop_point"] = polygon.mean(0).astype(np.float32)
        out["stop_direction"] = direction.astype(np.float32)
        out["stop_distance"] = np.float32(arclength)
        if state in state_names:
            out["traffic_light_state"].fill(state_names.index(state))
            out["traffic_light_state_valid"] = line
        out["red_stop_valid"] = np.float32(state == "red")
        return out


def offroad_distance_field(drivable: np.ndarray, bev: BevParams) -> np.ndarray:
    """把可行驶掩码转成单侧距离场：可行驶处为 0，越界处为到可行驶区域的距离（米）。

    输入可先扣除可见 box 占用；距离场让落在道路外或占用内的航点都获得连续回拉信号。
    """
    check_drivable_mask(drivable, bev)
    outside = np.ascontiguousarray(drivable < 0.5, dtype=np.uint8)
    if not bool((outside == 0).any()):
        diagonal_m = math.hypot(bev.x_max - bev.x_min, bev.y_max - bev.y_min)
        return np.full((bev.height, bev.width), diagonal_m, dtype=np.float32)

    distance_px = cv2.distanceTransform(outside, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    x_cell_m = (bev.x_max - bev.x_min) / bev.height
    y_cell_m = (bev.y_max - bev.y_min) / bev.width
    meters_per_pixel = 0.5 * (x_cell_m + y_cell_m)
    return (distance_px * meters_per_pixel).astype(np.float32)


def _rasterize_lane(class_map, direction, rows, cols, vectors, class_id, thickness):
    """把一条稠密折线的类别与逐点方向栅格化；同像素方向先平均，线宽邻域再作局部平均。"""
    h, w = class_map.shape
    rows = np.rint(rows).astype(np.int64)
    cols = np.rint(cols).astype(np.int64)
    valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    if not bool(valid.any()):
        return
    flat = rows[valid] * w + cols[valid]
    unique, inverse = np.unique(flat, return_inverse=True)
    sums = np.zeros((len(unique), 2), dtype=np.float32)
    np.add.at(sums, inverse, vectors[valid].astype(np.float32))
    sums /= np.linalg.norm(sums, axis=1, keepdims=True).clip(1e-6)

    seed = np.zeros((h, w), dtype=np.float32)
    vx = np.zeros_like(seed)
    vy = np.zeros_like(seed)
    seed.flat[unique] = 1.0
    vx.flat[unique] = sums[:, 0]
    vy.flat[unique] = sums[:, 1]
    kernel = np.ones((thickness, thickness), dtype=np.float32)
    weight = cv2.filter2D(seed, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    mask = weight > 0
    local_x = cv2.filter2D(vx, -1, kernel, borderType=cv2.BORDER_CONSTANT)[mask] / weight[mask]
    local_y = cv2.filter2D(vy, -1, kernel, borderType=cv2.BORDER_CONSTANT)[mask] / weight[mask]
    norm = np.hypot(local_x, local_y).clip(1e-6)
    class_map[mask] = class_id
    direction[0, mask] = local_x / norm
    direction[1, mask] = local_y / norm


def _empty_traffic_control(bev):
    """无相关停止线时的稳定张量契约；状态值由 valid 掩码屏蔽。"""
    shape = (bev.height, bev.width)
    return {
        "stop_line": np.zeros(shape, dtype=np.float32),
        "traffic_light_state": np.zeros(shape, dtype=np.int64),
        "traffic_light_state_valid": np.zeros(shape, dtype=np.float32),
        "stop_point": np.zeros(2, dtype=np.float32),
        "stop_direction": np.zeros(2, dtype=np.float32),
        "stop_distance": np.float32(0.0),
        "red_stop_valid": np.float32(0.0),
    }


def _select_relevant_control(control_polygons, actor_ids, world_to_ego_matrix,
                             route, route_mask, bev):
    """筛出前方、BEV 内且与路线走廊相交的控制区，并返回沿路线最近者。"""
    candidates = []
    for polygon_world, actor_id in zip(control_polygons, actor_ids):
        polygon = transform_points(polygon_world, world_to_ego_matrix)[:, :2]
        center = polygon.mean(0)
        if center[0] <= max(bev.x_min, 0.0) or not _polygon_in_bev(polygon, bev):
            continue
        if not bool(np.any(route_mask & _polygon_mask(polygon, bev))):
            continue
        arclength, direction = _route_projection(center, route)
        candidates.append((arclength, polygon, direction, actor_id))
    return min(candidates, key=lambda item: item[0]) if candidates else None


def _clean_route(route_xy):
    """去除非法点与连续重复点，避免静止帧产生零长度路线段。"""
    route = np.asarray(route_xy, dtype=np.float64)
    if route.ndim != 2 or route.shape[1] != 2:
        return np.empty((0, 2), dtype=np.float64)
    route = route[np.isfinite(route).all(1)]
    if len(route) < 2:
        return route
    keep = np.r_[True, np.linalg.norm(np.diff(route, axis=0), axis=1) > 1e-3]
    return route[keep]


def _route_mask(route, bev, corridor_m):
    """把路线折线膨胀成走廊；相关性在最终监督分辨率上做，和停止区栅格契约一致。"""
    mask = np.zeros((bev.height, bev.width), dtype=np.uint8)
    rows, cols = ego_xy_to_pixel(route, bev)
    points = np.stack((cols, rows), axis=1).round().astype(np.int32).reshape(-1, 1, 2)
    pixels_per_m = 0.5 * (bev.height / (bev.x_max - bev.x_min)
                          + bev.width / (bev.y_max - bev.y_min))
    thickness = max(int(round(2.0 * corridor_m * pixels_per_m)), 1)
    cv2.polylines(mask, [points], isClosed=False, color=1, thickness=thickness)
    return mask.astype(bool)


def _polygon_mask(polygon, bev):
    """ego 米制多边形栅格化为二值掩码。"""
    mask = np.zeros((bev.height, bev.width), dtype=np.uint8)
    rows, cols = ego_xy_to_pixel(polygon, bev)
    points = np.stack((cols, rows), axis=1).round().astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [points], color=1)
    return mask.astype(bool)


def _expanded_polygon_mask(polygon, bev, expand_m):
    """栅格化并按米制半径膨胀停止区，给细线监督提供稳定正样本。"""
    mask = _polygon_mask(polygon, bev).astype(np.uint8)
    if expand_m <= 0:
        return mask
    pixels_per_m = 0.5 * (bev.height / (bev.x_max - bev.x_min)
                          + bev.width / (bev.y_max - bev.y_min))
    radius = max(int(round(expand_m * pixels_per_m)), 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask, kernel)


def _polygon_in_bev(polygon, bev):
    """多边形包围盒与 BEV 有交集。"""
    return (polygon[:, 0].max() >= bev.x_min and polygon[:, 0].min() <= bev.x_max
            and polygon[:, 1].max() >= bev.y_min and polygon[:, 1].min() <= bev.y_max)


def _route_projection(point, route):
    """返回点在路线最近投影处的累计弧长与有向单位切向。"""
    starts, vectors = route[:-1], np.diff(route, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    unit = vectors / lengths[:, None]
    along = np.clip(((point - starts) * unit).sum(1), 0.0, lengths)
    closest = starts + along[:, None] * unit
    index = int(np.argmin(np.linalg.norm(closest - point, axis=1)))
    return lengths[:index].sum() + along[index], unit[index]


def _match_control_actors(parent_xy, traffic_lights, radius_m):
    """按地图 ParentActor 世界坐标匹配场景交通灯；未匹配项记为 -1。"""
    if len(parent_xy) == 0 or not traffic_lights:
        return np.full(len(parent_xy), -1, dtype=np.int64)
    actor_xy = np.array([item["transform"][:2] for item in traffic_lights], dtype=np.float64)
    actor_ids = np.array([item["id"] for item in traffic_lights], dtype=np.int64)
    distances = np.linalg.norm(parent_xy[:, None] - actor_xy[None], axis=-1)
    nearest = distances.argmin(1)
    return np.where(distances[np.arange(len(parent_xy)), nearest] <= radius_m,
                    actor_ids[nearest], -1)


def _parse_traffic_controls(arr):
    """解析交通灯触发区世界多边形与父 actor 位置。"""
    controls = []
    for row in arr:
        lanes = row[1]
        if not isinstance(lanes, dict):
            continue
        for item in lanes.get("Trigger_Volumes", []):
            if not isinstance(item, dict) or item.get("Type") != "TrafficLight":
                continue
            polygon = np.asarray(item.get("Points", []), dtype=np.float64)
            parent = np.asarray(item.get("ParentActor_Location", []), dtype=np.float64)
            if polygon.ndim == 2 and polygon.shape[0] >= 3 and polygon.shape[1] == 3 \
                    and parent.shape == (3,):
                controls.append((polygon, parent[:2]))
    return controls


def _parse_lanes(arr):
    """解析世界系折线、逐点有向切向量与 Type，跳过 Trigger_Volumes。"""
    lanes_out = []
    for row in arr:
        lanes = row[1]
        if not isinstance(lanes, dict):
            continue
        for lane_id, segs in lanes.items():
            if lane_id == "Trigger_Volumes" or not isinstance(segs, list):
                continue
            for seg in segs:
                pts = seg.get("Points") if isinstance(seg, dict) else None
                if not pts:
                    continue
                xyz = np.array([point[0] for point in pts], dtype=np.float64)  # point=((x,y,z),(rpy))
                if xyz.ndim == 2 and xyz.shape[1] == 3 and len(xyz) >= 2:
                    yaw = np.deg2rad(np.array([point[1][2] for point in pts], dtype=np.float64))
                    vectors = np.stack((np.cos(yaw), np.sin(yaw)), axis=1)
                    lanes_out.append((xyz, vectors, str(seg.get("Type", ""))))
    return lanes_out
