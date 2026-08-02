"""重建 Mesh 的静态、动态与轨迹渲染公开 API。

模块: vis/reconstructed_mesh_vis/render/__init__.py
依赖: vis.reconstructed_mesh_vis.render.render
读取配置: —（配置由公开接口的 cfg 入参提供）
对外接口:
    - RenderState
    - render_static_mesh(data, state, cfg, bev_cfg) -> open3d.geometry.TriangleMesh
    - render_dynamic_mesh(data, state, cfg, bev_cfg) -> open3d.geometry.TriangleMesh
    - render_trajectories(data) -> open3d.geometry.LineSet
"""

from vis.reconstructed_mesh_vis.render.render import (
    RenderState,
    render_dynamic_mesh,
    render_static_mesh,
    render_trajectories,
)

__all__ = ["RenderState", "render_static_mesh", "render_dynamic_mesh",
           "render_trajectories"]
