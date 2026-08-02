"""融合重建点云 Open3D 可视化包：读取统一 PT、分层着色并交互浏览轨迹。

模块: vis/reconstructed_pointcloud_vis/__init__.py
依赖: vis.reconstructed_pointcloud_vis.reader/render/viewer
读取配置: —
对外接口:
    - FusionPointcloud
    - list_pointclouds
    - resolve_pointcloud
    - RenderState
    - render_pointcloud
    - render_trajectories
    - current_bev_mask
    - current_bev_center
    - PointcloudViewer
"""

from .reader import FusionPointcloud, list_pointclouds, resolve_pointcloud
from .render import (
    RenderState,
    current_bev_center,
    current_bev_mask,
    render_pointcloud,
    render_trajectories,
)
from .viewer import PointcloudViewer

__all__ = [
    "FusionPointcloud", "list_pointclouds", "resolve_pointcloud", "RenderState",
    "render_pointcloud", "render_trajectories", "current_bev_mask", "current_bev_center",
    "PointcloudViewer",
]
