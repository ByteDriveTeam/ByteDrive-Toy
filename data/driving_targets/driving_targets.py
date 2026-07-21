"""驾驶监督目标编码（numpy/OpenCV）：BEV/轨迹/三场、可见运动占用及八类多标签行为。

模块: data/driving_targets/driving_targets.py
依赖: numpy, cv2, math, vis.data_vis.geometry,
      data.driving_targets.checks.driving_targets_checks
读取配置: —（BEV 量程/分辨率/视场/K/σ 等由调用方以参数传入，来源 config.model.driving 与 config.data.driving）
对外接口:
    - BevParams(x_min,x_max,y_min,y_max,height,width)                     # BEV 栅格几何
    - bev_cell_centers(bev) -> (H,W,2)                                    # 每 cell 中心 ego xy
    - ego_xy_to_pixel(xy, bev) -> (rows, cols)                           # ego xy → BEV 像素(行,列)
    - inview_mask(bev, fov_deg) -> (H,W) float32                         # 前向视场内=1
    - speed_accelerations(world_velocities, sim_times) -> (F,) float32   # 逐帧标量速度加速度
    - trajectory_targets(future_poses6, current_pose6, num_waypoints) -> (waypoints(K,2), valid(K))
    - behavior_targets(...) -> (8,) float32                              # 固定顺序的行为多热标签
    - risk_field(depth_m, intrinsics4, extrinsic6, bev, fov_deg, depth_max_m) -> (H,W) float32  # 包络外=风险
    - visible_moving_box_occupancy(...) -> (H,W) float32                  # 深度筛选后的运动 box BEV 占用
    - distribution_field(waypoints, valid, bev, sigma_m) -> (H,W) float32           # GT 航点高斯软占据
说明: BEV 为 ego 前向单目：行(H)沿 x 前向、列(W)沿 y 右向；自车位于下方中心（x=x_min 在最下行）。几何变换复用
      vis.data_vis.geometry（CARLA 左手系，已测），故 data 侧不重复实现投影。风险场把全部深度像素反投影到 ego
      BEV 得表面点云，按方位角取最大观测距离作外缘线包络，cell 落在其方位包络之外（更远）即遮挡/未观测→风险。
      可行驶区域只考虑可运动类别 vehicle/pedestrian，二者不分类、统一视为占用；先筛出与 BEV 相交的 3D 框，
      再把 GT 深度像素反投影，框内深度点不少于配置阈值（默认 10 像素）才栅格化其 BEV 足迹。ego 与场景级
      traffic_sign/traffic_light/pole/static 均不参与。
      行为标签固定为「障碍停车、红灯停车、加速、直行、左转、右转、减速、静止」：静止依据速度，纵向行为依据
      帧间速度加速度，转向依据最远有效航点方位；障碍停车还要求前方本车道走廊内有动态 Agent。新数据管线可
      直接传入「当前路线相关停止线为红灯」以提前激活红灯停车；未传入时兼容旧的静止+可见灯判定。
      各类互不排斥，可同时为 1。
"""

from __future__ import annotations

import functools
import math
from collections import namedtuple

import cv2
import numpy as np

from data.driving_targets.checks.driving_targets_checks import (
    check_behavior_inputs,
    check_bev_params,
    check_depth_map,
    check_motion_sequence,
    check_visible_moving_box_inputs,
)
from vis.data_vis.geometry import (
    bbox_corners,
    intrinsic_matrix,
    project_points,
    transform_matrix,
    transform_points,
    world_to_camera,
    world_to_ego,
)


__all__ = [
    "BEHAVIOR_CLASSES", "BehaviorParams", "BevParams", "bev_cell_centers", "ego_xy_to_pixel",
    "inview_mask", "speed_accelerations", "trajectory_targets", "behavior_targets",
    "risk_field", "visible_moving_box_occupancy", "distribution_field",
]

BevParams = namedtuple("BevParams", ["x_min", "x_max", "y_min", "y_max", "height", "width"])
BehaviorParams = namedtuple("BehaviorParams", [
    "stationary_speed_mps", "acceleration_threshold_mps2", "turn_angle_deg", "lane_half_width_m",
    "traffic_light_semantic_tag", "traffic_light_match_radius_m", "traffic_light_seg_margin_px",
    "traffic_light_min_pixels",
])

# 下标即模型输出/训练标签的稳定语义契约，顺序与用户给出的八类完全一致。
BEHAVIOR_CLASSES = (
    "obstacle_stop", "red_light_stop", "accelerating", "straight",
    "left_turn", "right_turn", "decelerating", "stationary",
)

# 采集契约中只有车辆与行人是可运动环境参与者；ego 和场景级静态框不得进入可行驶占用监督。
_MOVING_BOX_SEMANTICS = frozenset(("vehicle", "pedestrian"))

# BEV 网格与图像像素网格是随几何常量不变的量，同一 dataset 里只有极少数取值；有界记忆化避免每样本
# 重复分配大数组，缓存条目数为常量、绝不随场景/样本增长（不重演历史无界内存问题）。
_GEOMETRY_CACHE_SIZE = 8


@functools.lru_cache(maxsize=_GEOMETRY_CACHE_SIZE)
def bev_cell_centers(bev: BevParams) -> np.ndarray:
    """每 BEV cell 中心的 ego 平面坐标 `(H, W, 2)`=(x, y)。

    行约定与 ego_xy_to_pixel 一致：行 0 = 远 x_max（图像上沿），末行 = 近 x_min（自车在下沿中心）。
    结果只依赖 bev 几何常量，故按 BevParams 有界记忆化并置为只读——调用方一律只读取、不就地改写。
    """
    check_bev_params(bev)
    x_cell = (bev.x_max - bev.x_min) / bev.height
    y_cell = (bev.y_max - bev.y_min) / bev.width
    xs = bev.x_max - (np.arange(bev.height) + 0.5) * x_cell        # 行 0 = 远、末行 = 近（自车在下沿）
    ys = bev.y_min + (np.arange(bev.width) + 0.5) * y_cell
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    centers = np.stack((gx, gy), axis=-1)
    centers.flags.writeable = False
    return centers


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


def speed_accelerations(world_velocities: np.ndarray, sim_times: np.ndarray) -> np.ndarray:
    """由逐帧世界速度与仿真时间计算标量速度加速度 `(F,)`。

    相邻帧先作一阶差分；内部帧取前后斜率均值，首尾使用单侧斜率。速度模长与坐标旋转无关，适合直接判定
    加速/减速。时间戳异常或重复的边对应斜率置零，避免产生无穷标签。
    """
    check_motion_sequence(world_velocities, sim_times)
    speeds = np.linalg.norm(world_velocities[:, :2], axis=1)
    if len(speeds) < 2:
        return np.zeros_like(speeds, dtype=np.float32)
    delta_t = np.diff(sim_times)
    slopes = np.divide(np.diff(speeds), delta_t, out=np.zeros_like(delta_t), where=delta_t > 0)
    accelerations = np.empty_like(speeds)
    accelerations[0], accelerations[-1] = slopes[0], slopes[-1]
    if len(speeds) > 2:
        accelerations[1:-1] = 0.5 * (slopes[:-1] + slopes[1:])
    return accelerations.astype(np.float32)


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


def behavior_targets(waypoints: np.ndarray, valid: np.ndarray, speed_mps: float,
                     acceleration_mps2: float, bboxes, traffic_lights, traffic_light_states,
                     static_bboxes, semantic: np.ndarray, current_pose6, intrinsics,
                     camera_extrinsic6, bev: BevParams, fov_deg: float,
                     params: BehaviorParams, red_light_relevant=None) -> np.ndarray:
    """生成固定顺序的八类行为多热标签 `(8,)`。

    标签顺序见 `BEHAVIOR_CLASSES`。转向三类按最远有效未来航点判定且至多激活一个；加速/减速由标量速度
    加速度判定；静止可与转向、停车原因同时存在。障碍停车仍仅在静止时按前方动态 Agent 判定；若调用方传入
    `red_light_relevant`，红灯停车表示相关停止线当前为红灯并可在减速阶段提前激活。未传入时保留旧的
    「静止 + 红灯 actor + Seg 可见性」判定，兼容独立调用方。
    """
    check_behavior_inputs(waypoints, valid, semantic)
    labels = np.zeros(len(BEHAVIOR_CLASSES), dtype=np.float32)
    stationary = speed_mps <= params.stationary_speed_mps
    labels[2] = float(acceleration_mps2 >= params.acceleration_threshold_mps2)
    labels[6] = float(acceleration_mps2 <= -params.acceleration_threshold_mps2)
    labels[7] = float(stationary)
    _set_direction_labels(labels, waypoints, valid, params.turn_angle_deg)

    if red_light_relevant is not None:
        # 新数据管线以「当前路线的第一条停止线为红灯」表达提前停车意图，不再等车辆静止后才给正标签。
        labels[1] = float(red_light_relevant)
    if stationary:
        labels[0] = float(_has_front_agent(
            bboxes, current_pose6, bev, params.lane_half_width_m))
        if red_light_relevant is None:
            labels[1] = float(_has_visible_red_light(
                traffic_lights, traffic_light_states, static_bboxes, semantic, current_pose6,
                intrinsics, camera_extrinsic6, bev, fov_deg, params))
    return labels


def _set_direction_labels(labels, waypoints, valid, turn_angle_deg):
    """按最远有效前向航点设置直行/左转/右转之一；CARLA ego 系 y 正向为右。"""
    idx = np.nonzero(valid > 0)[0]
    if idx.size == 0:
        return
    x, y = waypoints[idx[-1]]
    if x <= 0:
        return
    angle_deg = math.degrees(math.atan2(float(y), float(x)))
    labels[3] = float(abs(angle_deg) <= turn_angle_deg)
    labels[4] = float(angle_deg < -turn_angle_deg)
    labels[5] = float(angle_deg > turn_angle_deg)


def _has_front_agent(bboxes, current_pose6, bev, lane_half_width_m):
    """动态 Agent 框在 ego-BEV 前方且与本车道横向走廊相交。"""
    agents = [box for box in bboxes if box.get("semantic") in ("vehicle", "pedestrian")]
    if not agents:
        return False
    w2e = world_to_ego(current_pose6)
    corners = np.stack([transform_points(bbox_corners(box), w2e) for box in agents])
    centers = transform_points(np.array([box["location"] for box in agents]), w2e)
    x_min, x_max = corners[:, :, 0].min(1), corners[:, :, 0].max(1)
    y_min, y_max = corners[:, :, 1].min(1), corners[:, :, 1].max(1)
    in_bev = ((centers[:, 0] > max(bev.x_min, 0.0)) & (centers[:, 0] <= bev.x_max)
              & (centers[:, 1] >= bev.y_min) & (centers[:, 1] <= bev.y_max))
    overlaps_lane = ((x_max > max(bev.x_min, 0.0)) & (x_min <= bev.x_max)
                     & (y_max >= -lane_half_width_m) & (y_min <= lane_half_width_m))
    return bool(np.any(in_bev & overlaps_lane))


def _has_visible_red_light(traffic_lights, states, static_bboxes, semantic, current_pose6,
                           intrinsics, camera_extrinsic6, bev, fov_deg, params):
    """红灯 actor 状态 + 静态框视场/BEV + 投影框 Seg 像素联合判定可见红灯。"""
    red_ids = {state["id"] for state in states if state.get("state") == "red"}
    red_lights = [light for light in traffic_lights if light.get("id") in red_ids]
    light_boxes = [box for box in static_bboxes if box.get("semantic") == "traffic_light"]
    if not red_lights or not light_boxes:
        return False

    red_xy = np.array([light["transform"][:2] for light in red_lights], dtype=np.float64)
    box_xy = np.array([box["location"][:2] for box in light_boxes], dtype=np.float64)
    match_distance = np.linalg.norm(box_xy[:, None] - red_xy[None], axis=-1).min(axis=1)
    matched = [box for box, distance in zip(light_boxes, match_distance)
               if distance <= params.traffic_light_match_radius_m]
    if not matched:
        return False

    centers_ego = transform_points(np.array([box["location"] for box in matched]), world_to_ego(current_pose6))
    half_fov = math.radians(fov_deg) * 0.5
    bearings = np.arctan2(centers_ego[:, 1], np.maximum(centers_ego[:, 0], 1e-3))
    in_bev_view = ((centers_ego[:, 0] > max(bev.x_min, 0.0)) & (centers_ego[:, 0] <= bev.x_max)
                   & (centers_ego[:, 1] >= bev.y_min) & (centers_ego[:, 1] <= bev.y_max)
                   & (np.abs(bearings) <= half_fov))
    candidates = [box for box, visible in zip(matched, in_bev_view) if visible]
    if not candidates:
        return False

    w2c = world_to_camera(current_pose6, camera_extrinsic6)
    k = intrinsic_matrix(intrinsics)
    return any(_bbox_hits_semantic(
        box, semantic, w2c, k, params.traffic_light_semantic_tag,
        params.traffic_light_seg_margin_px, params.traffic_light_min_pixels) for box in candidates)


def _bbox_hits_semantic(box, semantic, w2c, k, semantic_tag, margin_px, min_pixels):
    """静态框投影矩形内命中足量指定 Seg 标签；完全在相机后方/画外则为假。"""
    uv, depth = project_points(bbox_corners(box), w2c, k)
    uv = uv[depth > 0]
    if len(uv) == 0:
        return False
    height, width = semantic.shape
    x0, y0 = np.floor(uv.min(0)).astype(np.int64) - margin_px
    x1, y1 = np.ceil(uv.max(0)).astype(np.int64) + margin_px
    x0, y0 = max(int(x0), 0), max(int(y0), 0)
    x1, y1 = min(int(x1), width - 1), min(int(y1), height - 1)
    if x0 > x1 or y0 > y1:
        return False
    return int(np.count_nonzero(semantic[y0:y1 + 1, x0:x1 + 1] == semantic_tag)) >= min_pixels


def visible_moving_box_occupancy(bboxes, depth_m: np.ndarray, intrinsics, current_pose6,
                                 camera_extrinsic6, bev: BevParams, depth_max_m: float,
                                 min_visible_pixels: int) -> np.ndarray:
    """把深度图确认可见的运动类别 3D box 栅格化为 BEV 占用 `(H,W)`。

    参数:
        bboxes: 世界系 3D 框；仅 vehicle/pedestrian 参与，二者统一作为占用。
        depth_m: 当前相机 GT 深度图（米）。
        intrinsics: 当前相机内参 dict。
        current_pose6: 当前主车世界位姿。
        camera_extrinsic6: 相机相对主车外参。
        bev: 输出 BEV 几何。
        depth_max_m: 有效监督深度上限。
        min_visible_pixels: 判为可见所需的最少框内深度像素数；配置保证不少于 10。
    返回:
        float32 二值占用图；1 表示当前相机可见 box 的地面足迹。
    """
    check_visible_moving_box_inputs(depth_m, min_visible_pixels)
    occupancy = np.zeros((bev.height, bev.width), dtype=np.uint8)
    boxes = [box for box in bboxes if box.get("semantic") in _MOVING_BOX_SEMANTICS]
    if not boxes:
        return occupancy.astype(np.float32)

    world_corners = np.stack([bbox_corners(box) for box in boxes])
    ego_corners = transform_points(
        world_corners.reshape(-1, 3), world_to_ego(current_pose6)).reshape(-1, 8, 3)
    relevant = _boxes_intersect_bev(ego_corners, bev)
    if not bool(relevant.any()):
        return occupancy.astype(np.float32)

    boxes = [box for box, keep in zip(boxes, relevant) if keep]
    ego_corners = ego_corners[relevant]
    world_corners = world_corners[relevant]
    camera_to_world = transform_matrix(current_pose6) @ transform_matrix(camera_extrinsic6)
    uv, camera_depth = project_points(
        world_corners.reshape(-1, 3), np.linalg.inv(camera_to_world),
        intrinsic_matrix(intrinsics))
    uv = uv.reshape(-1, 8, 2)
    camera_depth = camera_depth.reshape(-1, 8)
    visible = np.array([
        _box_matches_depth(box, box_uv, box_depth, depth_m, intrinsics, camera_to_world,
                           depth_max_m, min_visible_pixels)
        for box, box_uv, box_depth in zip(boxes, uv, camera_depth)
    ])

    for corners in ego_corners[visible]:
        rows, cols = ego_xy_to_pixel(corners[:, :2], bev)
        polygon = cv2.convexHull(np.stack((cols, rows), axis=1).round().astype(np.int32))
        cv2.fillConvexPoly(occupancy, polygon, 1)
    return occupancy.astype(np.float32)


def _boxes_intersect_bev(ego_corners: np.ndarray, bev: BevParams) -> np.ndarray:
    """以框足迹外接范围快速排除 BEV 外 box，避免对全场景静态框逐个做图像检查。"""
    x, y = ego_corners[:, :, 0], ego_corners[:, :, 1]
    return ((x.max(1) >= bev.x_min) & (x.min(1) <= bev.x_max)
            & (y.max(1) >= bev.y_min) & (y.min(1) <= bev.y_max))


def _box_matches_depth(box, uv: np.ndarray, camera_depth: np.ndarray, depth_m: np.ndarray,
                       intrinsics, camera_to_world: np.ndarray, depth_max_m: float,
                       min_pixels: int) -> bool:
    """投影凸包内有足量 GT 深度命中逐像素射线与框的交段，才认定框至少部分可见。"""
    front = camera_depth > 0
    if np.count_nonzero(front) < 3:
        return False
    height, width = depth_m.shape
    points = uv[front].copy()
    if (points[:, 0].max() < 0 or points[:, 0].min() > width - 1
            or points[:, 1].max() < 0 or points[:, 1].min() > height - 1):
        return False
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    polygon = cv2.convexHull(points.round().astype(np.int32))
    x, y, w, h = cv2.boundingRect(polygon)
    if w == 0 or h == 0:
        return False

    local_polygon = polygon - np.array([[[x, y]]], dtype=np.int32)
    projected = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(projected, local_polygon, 1)
    local_y, local_x = np.nonzero(projected)
    pixel_x, pixel_y = local_x + x, local_y + y
    observed = depth_m[pixel_y, pixel_x]
    rays = np.stack((
        np.ones_like(pixel_x),
        (pixel_x - intrinsics["cx"]) / intrinsics["fx"],
        -(pixel_y - intrinsics["cy"]) / intrinsics["fy"],
    ), axis=1)
    box_pose = [*box["location"], *box["rotation"]]
    camera_to_box = np.linalg.inv(transform_matrix(box_pose)) @ camera_to_world
    near, far, intersects = _ray_box_depth_interval(rays, camera_to_box, box["extent"])
    matches = (intersects & (observed > 0) & (observed < depth_max_m)
               & (observed >= near) & (observed <= far))
    return int(np.count_nonzero(matches)) >= min_pixels


def _ray_box_depth_interval(rays: np.ndarray, camera_to_box: np.ndarray, extent):
    """以相机前向深度为参数，批量求射线在定向框局部坐标中的进入/离开深度。"""
    origin = camera_to_box[:3, 3]
    directions = rays @ camera_to_box[:3, :3].T
    extent = np.asarray(extent, dtype=np.float64)
    parallel = np.abs(directions) < np.finfo(np.float64).eps
    safe_directions = np.where(parallel, 1.0, directions)
    t0 = (-extent - origin) / safe_directions
    t1 = (extent - origin) / safe_directions
    axis_near = np.where(parallel, -np.inf, np.minimum(t0, t1))
    axis_far = np.where(parallel, np.inf, np.maximum(t0, t1))
    near, far = axis_near.max(1), axis_far.min(1)
    parallel_outside = np.any(parallel & (np.abs(origin) > extent), axis=1)
    intersects = ~parallel_outside & (far >= np.maximum(near, 0.0))
    return near, far, intersects


# 风险场方位角分箱数（envelope 角分辨率；与相机水平像素数同量级足以贴合外缘线）
_RISK_BEARING_BINS = 256
_RISK_MIN_DEPTH_M = 0.1


@functools.lru_cache(maxsize=_GEOMETRY_CACHE_SIZE)
def _image_pixel_grid(hc: int, wc: int):
    """像平面 (行, 列) 坐标网格 `(vv, uu)`，随图像尺寸恒定，有界记忆化并置只读供反投影复用。"""
    vv, uu = np.meshgrid(np.arange(hc), np.arange(wc), indexing="ij")
    vv.flags.writeable = False
    uu.flags.writeable = False
    return vv, uu


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
    vv, uu = _image_pixel_grid(hc, wc)          # 像素坐标网格随图像尺寸恒定，有界记忆化
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
