"""Open3D Mesh 全局/自车 BEV 交互查看器公开 API。

模块: vis/reconstructed_mesh_vis/viewer/__init__.py
依赖: vis.reconstructed_mesh_vis.viewer.viewer
读取配置: —（配置由 MeshViewer 的 cfg 入参提供）
对外接口:
    - MeshViewer(data, cfg, bev_cfg).run() -> None
"""

from vis.reconstructed_mesh_vis.viewer.viewer import MeshViewer

__all__ = ["MeshViewer"]
