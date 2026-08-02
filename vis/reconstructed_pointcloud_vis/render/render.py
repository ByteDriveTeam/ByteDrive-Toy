"""Open3D 重建点云渲染：图层筛选、分层下采样、着色与 actor 轨迹生成。

模块: vis/reconstructed_pointcloud_vis/render/render.py
依赖: dataclasses, numpy, open3d, vis.reconstructed_pointcloud_vis.render.checks
读取配置: reconstructed_pointcloud_vis.max_static_points/max_dynamic_points、
          static_rgb/dynamic_rgb；model.driving.bev.x_min_m/x_max_m/y_min_m/y_max_m
对外接口:
    - RenderState                              # 当前图层、帧与着色状态
    - render_pointcloud(data, state, cfg, bev_cfg) -> open3d.geometry.PointCloud
    - render_trajectories(data) -> open3d.geometry.LineSet
    - current_bev_mask(points, ego_pose, bev_cfg) -> numpy.ndarray
    - current_bev_center(ego_pose, bev_cfg) -> numpy.ndarray
    - semantic_name(tag) -> str
说明: 语义枚举严格对应项目 CARLA 0.9.15 CityObjectLabel 0..28；颜色为稳定显示常量。
"""

from dataclasses import dataclass

import numpy as np
import open3d as o3d

from vis.reconstructed_pointcloud_vis.render.checks import check_render_state


SEMANTIC_NAMES = (
    "NONE", "Roads", "Sidewalks", "Buildings", "Walls", "Fences", "Poles",
    "TrafficLight", "TrafficSigns", "Vegetation", "Terrain", "Sky", "Pedestrians",
    "Rider", "Car", "Truck", "Bus", "Train", "Motorcycle", "Bicycle", "Static",
    "Dynamic", "Other", "Water", "RoadLines", "Ground", "Bridge", "RailTrack",
    "GuardRail",
)

_SEMANTIC_RGB = np.asarray((
    (0, 0, 0), (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
    (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
    (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
    (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100),
    (0, 0, 230), (119, 11, 32), (110, 110, 110), (255, 190, 0), (81, 0, 81),
    (45, 60, 150), (255, 255, 255), (81, 0, 81), (150, 100, 100),
    (230, 150, 140), (180, 165, 180),
), dtype=np.float64) / 255.0

_COLOR_MODES = ("semantic", "source", "actor", "height")


@dataclass
class RenderState:
    """描述一次点云渲染所需的交互状态。"""

    show_static: bool
    show_dynamic: bool
    show_trajectory: bool
    all_dynamic_frames: bool
    frame_index: int
    color_mode: str
    spatial_scope: str


def semantic_name(tag):
    """返回 CARLA 0.9.15 语义标签名，未知值返回 ``Unknown-<值>``。"""
    value = int(tag)
    return SEMANTIC_NAMES[value] if 0 <= value < len(SEMANTIC_NAMES) \
        else "Unknown-{}".format(value)


def render_pointcloud(data, state, cfg, bev_cfg):
    """把当前图层和帧构造成 Open3D PointCloud。

    静态与动态分别限制点数，避免静态地图吞掉动态对象的显示预算。
    """
    check_render_state(state, data)
    static_indices = np.arange(data.num_static, dtype=np.int64)
    if state.spatial_scope == "bev":
        static_indices = static_indices[current_bev_mask(
            data.static_xyz, data.ego_pose[state.frame_index], bev_cfg)]
    static_indices = _uniform_indices(static_indices, cfg.max_static_points) \
        if state.show_static else np.empty(0, dtype=np.int64)
    static_xyz = data.static_xyz[static_indices]
    static_tags = data.static_obj_tag[static_indices]
    dynamic_xyz, dynamic_tags, dynamic_actors = _placed_dynamic(data, state, cfg)
    if state.spatial_scope == "bev" and len(dynamic_xyz):
        keep = current_bev_mask(dynamic_xyz, data.ego_pose[state.frame_index], bev_cfg)
        dynamic_xyz, dynamic_tags, dynamic_actors = (
            dynamic_xyz[keep], dynamic_tags[keep], dynamic_actors[keep])
    points = np.concatenate((static_xyz, dynamic_xyz)).astype(np.float64, copy=False)
    tags = np.concatenate((static_tags, dynamic_tags))
    sources = np.concatenate((
        np.zeros(len(static_xyz), dtype=np.uint8),
        np.ones(len(dynamic_xyz), dtype=np.uint8),
    ))
    actors = np.concatenate((
        np.full(len(static_xyz), -1, dtype=np.int64), dynamic_actors,
    ))
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(
        _point_colors(tags, sources, actors, points, data, state.color_mode, cfg))
    return cloud


def render_trajectories(data):
    """直接由对象逐帧 Box 位姿生成完整世界坐标轨迹线。"""
    cloud = o3d.geometry.LineSet()
    if not data.num_poses:
        return cloud
    keys = (data.pose_object_index.astype(np.int64) << 32) \
        | data.pose_frame_index.astype(np.int64)
    order = np.argsort(keys)
    object_index = data.pose_object_index[order]
    centers = data.pose_transform[order, :3].astype(np.float64)
    connected = object_index[1:] == object_index[:-1]
    starts = np.flatnonzero(connected)
    lines = np.column_stack((starts, starts + 1)).astype(np.int32)
    cloud.points = o3d.utility.Vector3dVector(centers)
    cloud.lines = o3d.utility.Vector2iVector(lines)
    actor_id = data.object_actor_id[object_index[starts]]
    cloud.colors = o3d.utility.Vector3dVector(_actor_colors(actor_id))
    return cloud


def _placed_dynamic(data, state, cfg):
    if not state.show_dynamic or not data.num_poses or not data.num_dynamic:
        return (np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.uint8),
                np.empty(0, dtype=np.int64))
    pose_indices = np.arange(data.num_poses, dtype=np.int64)
    if not state.all_dynamic_frames:
        pose_indices = pose_indices[data.pose_frame_index == state.frame_index]
    object_index = data.pose_object_index[pose_indices]
    counts = np.diff(data.object_point_offsets)[object_index]
    total = int(counts.sum())
    if not total:
        return (np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.uint8),
                np.empty(0, dtype=np.int64))
    flat = np.arange(total, dtype=np.int64) if total <= cfg.max_dynamic_points \
        else np.linspace(0, total - 1, cfg.max_dynamic_points, dtype=np.int64)
    ends = np.cumsum(counts)
    selected_pose = np.searchsorted(ends, flat, side="right")
    starts = ends - counts
    selected_object = object_index[selected_pose]
    point_index = data.object_point_offsets[selected_object] \
        + flat - starts[selected_pose]
    transforms = data.pose_transform[pose_indices[selected_pose]]
    xyz = _transform_local(data.dynamic_xyz_local[point_index], transforms)
    return xyz, data.dynamic_obj_tag[point_index], data.object_actor_id[selected_object]


def _transform_local(points, transforms):
    radians = np.deg2rad(transforms[:, 3:6])
    cr, cp, cy = np.cos(radians).T
    sr, sp, sy = np.sin(radians).T
    rotations = np.empty((len(transforms), 3, 3), dtype=np.float32)
    rotations[:, 0, :] = np.column_stack((
        cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr))
    rotations[:, 1, :] = np.column_stack((
        sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr))
    rotations[:, 2, :] = np.column_stack((sp, -cp * sr, cp * cr))
    return np.einsum("nij,nj->ni", rotations, points) + transforms[:, :3]


def current_bev_mask(points, ego_pose, bev_cfg):
    """返回世界点落入当前自车朝向 BEV 矩形范围的布尔掩码。"""
    delta = np.asarray(points)[:, :2] - np.asarray(ego_pose)[:2]
    yaw = np.deg2rad(float(ego_pose[5]))
    cosine, sine = np.cos(yaw), np.sin(yaw)
    local_x = delta[:, 0] * cosine + delta[:, 1] * sine
    local_y = -delta[:, 0] * sine + delta[:, 1] * cosine
    return (local_x >= bev_cfg.x_min_m) & (local_x <= bev_cfg.x_max_m) \
        & (local_y >= bev_cfg.y_min_m) & (local_y <= bev_cfg.y_max_m)


def current_bev_center(ego_pose, bev_cfg):
    """返回当前自车 BEV 矩形中心的世界坐标。"""
    local_x = (bev_cfg.x_min_m + bev_cfg.x_max_m) * 0.5
    local_y = (bev_cfg.y_min_m + bev_cfg.y_max_m) * 0.5
    yaw = np.deg2rad(float(ego_pose[5]))
    cosine, sine = np.cos(yaw), np.sin(yaw)
    return np.asarray((
        ego_pose[0] + cosine * local_x - sine * local_y,
        ego_pose[1] + sine * local_x + cosine * local_y,
        ego_pose[2],
    ), dtype=np.float64)


def _uniform_indices(indices, limit):
    if len(indices) <= limit:
        return indices
    return indices[np.linspace(0, len(indices) - 1, limit, dtype=np.int64)]


def _point_colors(tags, sources, actors, points, data, mode, cfg):
    if not len(points):
        return np.empty((0, 3), dtype=np.float64)
    if mode == "semantic":
        semantic = tags.astype(np.int64)
        colors = np.full((len(semantic), 3), (1.0, 0.0, 1.0), dtype=np.float64)
        known = semantic < len(_SEMANTIC_RGB)
        colors[known] = _SEMANTIC_RGB[semantic[known]]
        return colors
    if mode == "source":
        palette = np.asarray((cfg.static_rgb, cfg.dynamic_rgb), dtype=np.float64) / 255.0
        return palette[sources]
    if mode == "actor":
        colors = _actor_colors(actors)
        colors[actors < 0] = np.asarray(cfg.static_rgb, dtype=np.float64) / 255.0
        return colors
    low, high = data.height_range
    scale = max(float(high - low), np.finfo(np.float32).eps)
    level = np.clip((points[:, 2] - low) / scale, 0.0, 1.0)
    return np.column_stack((level, 1.0 - np.abs(2.0 * level - 1.0), 1.0 - level))


def _actor_colors(actor_ids):
    ids = np.asarray(actor_ids, dtype=np.float64)
    hue = np.mod(ids * 0.6180339887498949, 1.0)
    sector = np.floor(hue * 6.0).astype(np.int64)
    fraction = hue * 6.0 - sector
    value, minimum = np.ones_like(hue), np.full_like(hue, 0.22)
    rising = minimum + (value - minimum) * fraction
    falling = value - (value - minimum) * fraction
    table = np.stack((
        np.column_stack((value, rising, minimum)),
        np.column_stack((falling, value, minimum)),
        np.column_stack((minimum, value, rising)),
        np.column_stack((minimum, falling, value)),
        np.column_stack((rising, minimum, value)),
        np.column_stack((value, minimum, falling)),
    ))
    return table[sector % 6, np.arange(len(ids))]


__all__ = [
    "RenderState", "SEMANTIC_NAMES", "render_pointcloud", "render_trajectories",
    "current_bev_mask", "current_bev_center", "semantic_name",
]
