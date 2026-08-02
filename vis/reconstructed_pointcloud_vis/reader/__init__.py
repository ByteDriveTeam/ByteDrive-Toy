"""融合重建点云读取模块公开 API。

模块: vis/reconstructed_pointcloud_vis/reader/__init__.py
依赖: vis.reconstructed_pointcloud_vis.reader.reader
读取配置: —
对外接口:
    - FusionPointcloud
    - list_pointclouds(root) -> list[Path]
    - resolve_pointcloud(spec, root) -> Path
"""

from .reader import FusionPointcloud, list_pointclouds, resolve_pointcloud

__all__ = ["FusionPointcloud", "list_pointclouds", "resolve_pointcloud"]
