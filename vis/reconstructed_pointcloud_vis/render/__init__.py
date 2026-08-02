"""Open3D 重建点云渲染模块公开 API。

模块: vis/reconstructed_pointcloud_vis/render/__init__.py
依赖: vis.reconstructed_pointcloud_vis.render.render
读取配置: —
对外接口:
    - RenderState
    - SEMANTIC_NAMES
    - render_pointcloud(data, state, cfg, bev_cfg) -> open3d.geometry.PointCloud
    - render_trajectories(data) -> open3d.geometry.LineSet
    - current_bev_mask(points, ego_pose, bev_cfg) -> numpy.ndarray
    - current_bev_center(ego_pose, bev_cfg) -> numpy.ndarray
    - semantic_name(tag) -> str
"""

from .render import (
    RenderState,
    SEMANTIC_NAMES,
    current_bev_center,
    current_bev_mask,
    render_pointcloud,
    render_trajectories,
    semantic_name,
)

__all__ = [
    "RenderState", "SEMANTIC_NAMES", "render_pointcloud", "render_trajectories",
    "current_bev_mask", "current_bev_center", "semantic_name",
]
