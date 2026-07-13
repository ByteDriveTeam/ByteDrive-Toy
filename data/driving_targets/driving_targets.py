"""驾驶监督目标编码（纯 numpy 函数）：BEV 几何、视场掩码、轨迹/扇区、风险场、轨迹分布场。

模块: data/driving_targets/driving_targets.py
依赖: numpy, math, vis.data_vis.geometry(transform_matrix/world_to_ego/transform_points),
      data.driving_targets.checks.driving_targets_checks
读取配置: —（BEV 量程/分辨率/视场/K/σ 等由调用方以参数传入，来源 config.model.driving 与 config.data.driving）
对外接口:
    - BevParams(x_min,x_max,y_min,y_max,height,width)                     # BEV 栅格几何
    - bev_cell_centers(bev) -> (H,W,2)                                    # 每 cell 中心 ego xy
    - ego_xy_to_pixel(xy, bev) -> (rows, cols)                           # ego xy → BEV 像素(行,列)
    - inview_mask(bev, fov_deg) -> (H,W) float32                         # 前向视场内=1
    - trajectory_targets(future_poses6, current_pose6, num_waypoints) -> (waypoints(K,2), valid(K))
    - sector_of(waypoints, valid, fov_deg, num_modes) -> int            # GT 所属前向扇区（-1=无效）
    - risk_field(depth_m, intrinsics4, extrinsic6, bev, fov_deg, depth_max_m) -> (H,W) float32  # 包络外=风险
    - distribution_field(waypoints, valid, bev, sigma_m) -> (H,W) float32           # GT 航点高斯软占据
说明: BEV 为 ego 前向单目：行(H)沿 x 前向、列(W)沿 y 右向；自车位于下方中心（x=x_min 在最下行）。几何变换复用
      vis.data_vis.geometry（CARLA 左手系，已测），故 data 侧不重复实现投影。风险场把全部深度像素反投影到 ego
      BEV 得表面点云，按方位角取最大观测距离作外缘线包络，cell 落在其方位包络之外（更远）即遮挡/未观测→风险。
      轨迹分布场对每个 GT 航点打各向同性高斯并取逐 cell 最大，得平滑正样本场。全部向量化（§9）。
"""

from __future__ import annotations

import math
from collections import namedtuple

import numpy as np

from data.driving_targets.checks.driving_targets_checks import check_bev_params, check_depth_map
from vis.data_vis.geometry import transform_matrix, transform_points, world_to_ego


__all__ = [
    "BevParams", "bev_cell_centers", "ego_xy_to_pixel", "inview_mask",
    "trajectory_targets", "sector_of", "risk_field", "distribution_field",
]

BevParams = namedtuple("BevParams", ["x_min", "x_max", "y_min", "y_max", "height", "width"])


def bev_cell_centers(bev: BevParams) -> np.ndarray:
    """每 BEV cell 中心的 ego 平面坐标 `(H, W, 2)`=(x, y)。

    行约定与 ego_xy_to_pixel 一致：行 0 = 远 x_max（图像上沿），末行 = 近 x_min（自车在下沿中心）。
    """
    check_bev_params(bev)
    x_cell = (bev.x_max - bev.x_min) / bev.height
    y_cell = (bev.y_max - bev.y_min) / bev.width
    xs = bev.x_max - (np.arange(bev.height) + 0.5) * x_cell        # 行 0 = 远、末行 = 近（自车在下沿）
    ys = bev.y_min + (np.arange(bev.width) + 0.5) * y_cell
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack((gx, gy), axis=-1)


def ego_xy_to_pixel(xy: np.ndarray, bev: BevParams):
    """ego 平面 xy → BEV 像素 (rows, cols)：x 前向映射到行（近在下沿），y 右向映射到列。"""
    x, y = xy[..., 0], xy[..., 1]
    rows = (1.0 - (x - bev.x_min) / (bev.x_max - bev.x_min)) * bev.height
    cols = (y - bev.y_min) / (bev.y_max - bev.y_min) * bev.width
    return rows, cols


def inview_mask(bev: BevParams, fov_deg: float) -> np.ndarray:
    """前向视场内掩码 `(H, W)`：x>0 且方位角 |atan2(y,x)| <= fov/2。"""
    centers = bev_cell_centers(bev)
    x, y = centers[..., 0], centers[..., 1]
    half = math.radians(fov_deg) * 0.5
    angle = np.arctan2(y, np.maximum(x, 1e-3))
    return ((x > 0) & (np.abs(angle) <= half)).astype(np.float32)


def trajectory_targets(future_poses6, current_pose6, num_waypoints: int):
    """未来 ego 世界位姿序列 → 当前 ego 系航点 `(K,2)` 与有效掩码 `(K,)`。

    参数:
        future_poses6: 未来帧 ego 世界位姿列表（每项 [x,y,z,roll,pitch,yaw]），最多取 num_waypoints 个。
        current_pose6: 当前帧 ego 世界位姿。
        num_waypoints: K；不足则补零并标记无效。
    """
    waypoints = np.zeros((num_waypoints, 2), dtype=np.float32)
    valid = np.zeros((num_waypoints,), dtype=np.float32)
    if not future_poses6:
        return waypoints, valid
    w2e = world_to_ego(current_pose6)
    future_xyz = np.array([p[:3] for p in future_poses6[:num_waypoints]], dtype=np.float64)
    ego_xyz = transform_points(future_xyz, w2e)                    # 未来位置搬到当前 ego 系
    n = ego_xyz.shape[0]
    waypoints[:n] = ego_xyz[:, :2].astype(np.float32)
    valid[:n] = 1.0
    return waypoints, valid


def sector_of(waypoints: np.ndarray, valid: np.ndarray, fov_deg: float, num_modes: int) -> int:
    """GT 轨迹所属前向扇区索引（按最远有效航点方位角分箱）；无有效前向航点返回 -1。"""
    idx = np.nonzero(valid > 0)[0]
    if idx.size == 0:
        return -1
    far = waypoints[idx[-1]]                                       # 最远有效航点
    x, y = float(far[0]), float(far[1])
    if x <= 0:
        return -1
    half = math.radians(fov_deg) * 0.5
    angle = math.atan2(y, x)
    if angle < -half or angle > half:
        return -1
    frac = (angle + half) / (2.0 * half)                          # [0,1]
    return int(min(max(int(frac * num_modes), 0), num_modes - 1))


# 风险场方位角分箱数（envelope 角分辨率；与相机水平像素数同量级足以贴合外缘线）
_RISK_BEARING_BINS = 256
_RISK_MIN_DEPTH_M = 0.1


def risk_field(depth_m: np.ndarray, intrinsics4, extrinsic6, bev: BevParams,
               fov_deg: float, depth_max_m: float) -> np.ndarray:
    """遮挡风险场 `(H, W)`：深度投影外缘线包络之外即风险。

    把全部有效深度像素反投影到 ego BEV 得可视表面点云，按方位角分箱取每个方位的最大观测距离，构成「恰好
    包裹所有平面投影点的外缘线」。BEV cell 位于其所在方位包络之外（更远）即为遮挡/未观测→风险=1；包络之内
    （已观测的自由/表面）为 0。排除超范围/天空像素（depth>=depth_max_m），使其不把包络推到无穷。无观测点的
    方位（含视场内被完全遮挡）包络为 0，其所有 cell 记为风险。
    """
    check_depth_map(depth_m)
    fx, fy, cx, cy = (float(v) for v in intrinsics4)
    half = math.radians(fov_deg) * 0.5
    ego_from_cam = transform_matrix(extrinsic6)

    # 反投影全部像素到 ego 平面（像平面系→传感器系→ego 系，与 vis.geometry.project_points 互逆）
    hc, wc = depth_m.shape
    vv, uu = np.meshgrid(np.arange(hc), np.arange(wc), indexing="ij")
    d = depth_m.astype(np.float64)
    sensor = np.stack([d, (uu - cx) / fx * d, -((vv - cy) / fy * d)], axis=-1).reshape(-1, 3)
    ego = transform_points(sensor, ego_from_cam)
    rng = np.hypot(ego[:, 0], ego[:, 1])
    bearing = np.arctan2(ego[:, 1], ego[:, 0])
    p_valid = (d.reshape(-1) > _RISK_MIN_DEPTH_M) & (d.reshape(-1) < depth_max_m) \
        & (ego[:, 0] > 0) & (np.abs(bearing) <= half)

    # 每个方位箱的最大观测距离 = 外缘线包络
    r_max = np.zeros(_RISK_BEARING_BINS)
    bins = _bearing_bin(bearing, half)
    np.maximum.at(r_max, bins[p_valid], rng[p_valid])

    # BEV cell：视场内且距离超过其方位包络即风险
    centers = bev_cell_centers(bev)
    cx_ego, cy_ego = centers[..., 0], centers[..., 1]
    c_rng = np.hypot(cx_ego, cy_ego)
    c_bear = np.arctan2(cy_ego, np.maximum(cx_ego, 1e-3))
    c_in = (cx_ego > 0) & (np.abs(c_bear) <= half)
    c_bin = _bearing_bin(c_bear, half)
    risk = c_in & (c_rng > r_max[c_bin])
    return risk.astype(np.float32)


def _bearing_bin(bearing: np.ndarray, half: float) -> np.ndarray:
    """把方位角 [-half, half] 量化到 [0, _RISK_BEARING_BINS)。"""
    frac = (bearing + half) / (2.0 * half)
    return np.clip((frac * _RISK_BEARING_BINS).astype(np.int64), 0, _RISK_BEARING_BINS - 1)


def distribution_field(waypoints: np.ndarray, valid: np.ndarray, bev: BevParams,
                       sigma_m: float) -> np.ndarray:
    """GT 航点高斯软占据场 `(H, W)`：每个有效航点打各向同性高斯，逐 cell 取最大。"""
    field = np.zeros((bev.height, bev.width), dtype=np.float32)
    idx = np.nonzero(valid > 0)[0]
    if idx.size == 0:
        return field
    centers = bev_cell_centers(bev)                              # (H,W,2)
    pts = waypoints[idx]                                         # (n,2)
    # 逐航点到全体 cell 的平方距离，高斯后取最大：(H,W,n) -> (H,W)
    diff = centers[:, :, None, :] - pts[None, None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)                         # (H,W,n)
    gauss = np.exp(-dist2 / (2.0 * sigma_m * sigma_m))
    return gauss.max(axis=-1).astype(np.float32)
