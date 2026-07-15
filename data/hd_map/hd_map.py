"""HD 地图：加载带类别/方向的车道折线，生成可行驶区域、独立道路线图及越界距离场。

模块: data/hd_map/hd_map.py
依赖: numpy, cv2, math, vis.data_vis.geometry(world_to_ego/transform_points),
      data.driving_targets(ego_xy_to_pixel/BevParams), data.hd_map.checks.hd_map_checks
读取配置: —（地图路径、BEV 几何、车道半宽由调用方传入，来源 config.data.driving 与 config.model.driving.bev）
对外接口:
    - HdMap(npz_path) -> HdMap
        .drivable_bev(ego_pose6, bev, lane_half_width_m) -> (H,W) float32   # 1=地图可行驶
        .lane_map_bev(ego_pose6, bev, line_width_m, type_to_class, unknown_class)
            -> (class_map[H,W], direction[2,H,W])
    - offroad_distance_field(drivable, bev) -> (H,W) float32                # 可行驶处=0，越界距离（米）
说明: npz 结构为 `arr[(road_id, {lane_id: [ {Points:[((x,y,z),(rpy),...)...], Type, Color, ...}, ... ]})]`；
      解析世界系 xyz、Type 与每点 yaw 对应的有向单位切向量。栅格化时按 ego 世界位姿把折线与方向搬到 ego 系
      （复用 vis.data_vis.geometry 的 CARLA
      变换，避免重复实现），投到 BEV 像素后用 cv2 粗线（宽 = 2·车道半宽）画出，其并集近似路面可行驶区域——
      稠密且不受语义投影稀疏之苦（用户口径）。逐折线用外接圆半径对 ego 做粗过滤，仅栅格化附近折线，控制每帧耗时。
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from data.driving_targets import BevParams, ego_xy_to_pixel
from data.hd_map.checks.hd_map_checks import check_drivable_mask, check_map_path, check_polylines
from vis.data_vis.geometry import transform_points, world_to_ego


__all__ = ["HdMap", "offroad_distance_field"]


class HdMap:
    """车道折线地图 → ego BEV 可行驶掩码与类别/方向道路线图栅格化器。"""

    def __init__(self, npz_path) -> None:
        check_map_path(npz_path)
        self._path = str(npz_path)
        lanes = _parse_lanes(np.load(self._path, allow_pickle=True)["arr"])
        self._polylines = [lane[0] for lane in lanes]
        self._directions = [lane[1] for lane in lanes]
        self._line_types = [lane[2] for lane in lanes]
        check_polylines(self._polylines, self._path)
        # 逐折线外接圆（世界系）：中心与半径，供按 ego 距离粗过滤
        self._centers = np.array([p[:, :2].mean(0) for p in self._polylines], dtype=np.float64)
        self._radii = np.array(
            [np.linalg.norm(p[:, :2] - c, axis=1).max() for p, c in zip(self._polylines, self._centers)],
            dtype=np.float64)

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
