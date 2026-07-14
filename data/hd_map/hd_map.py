"""HD 地图：加载车道折线，生成 BEV 可行驶掩码及越界距离场。

模块: data/hd_map/hd_map.py
依赖: numpy, cv2, math, vis.data_vis.geometry(world_to_ego/transform_points),
      data.driving_targets(ego_xy_to_pixel/BevParams), data.hd_map.checks.hd_map_checks
读取配置: —（地图路径、BEV 几何、车道半宽由调用方传入，来源 config.data.driving 与 config.model.driving.bev）
对外接口:
    - HdMap(npz_path) -> HdMap
        .drivable_bev(ego_pose6, bev, lane_half_width_m) -> (H,W) float32   # 1=可行驶
    - offroad_distance_field(drivable, bev) -> (H,W) float32                # 道路内=0，越界距离（米）
说明: npz 结构为 `arr[(road_id, {lane_id: [ {Points:[((x,y,z),(rpy))...], Type, Color, ...}, ... ]})]`；解析出
      全部车道折线的世界系 xyz。栅格化时按 ego 世界位姿把折线搬到 ego 系（复用 vis.data_vis.geometry 的 CARLA
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
    """车道折线地图 → ego BEV 可行驶掩码栅格化器。"""

    def __init__(self, npz_path) -> None:
        check_map_path(npz_path)
        self._path = str(npz_path)
        self._polylines = _parse_polylines(np.load(self._path, allow_pickle=True)["arr"])
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


def offroad_distance_field(drivable: np.ndarray, bev: BevParams) -> np.ndarray:
    """把 HDMap 可行驶掩码转成单侧距离场：道路内为 0，越界处为到道路的距离（米）。

    二值掩码只有边缘附近存在空间梯度；距离场让落在道路外的轨迹航点无论离边界多远都能获得连续回拉信号。
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


def _parse_polylines(arr):
    """把 npz arr 解析为世界系车道折线列表（每条 [P,3] xyz），跳过 Trigger_Volumes 等非车道项。"""
    polylines = []
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
                    polylines.append(xyz)
    return polylines
