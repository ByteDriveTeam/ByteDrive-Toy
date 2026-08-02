"""水密 Mesh Open3D 可视化包：读取统一 PT 并按逐帧位姿组合静动态几何。

模块: vis/reconstructed_mesh_vis/__init__.py
依赖: vis.reconstructed_mesh_vis.reader/render/viewer
读取配置: —
对外接口:
    - ReconstructedMesh(path)
    - RenderState
    - MeshViewer(data, cfg).run() -> None
"""

from vis.reconstructed_mesh_vis.reader import ReconstructedMesh
from vis.reconstructed_mesh_vis.render import RenderState
from vis.reconstructed_mesh_vis.viewer import MeshViewer

__all__ = ["ReconstructedMesh", "RenderState", "MeshViewer"]
