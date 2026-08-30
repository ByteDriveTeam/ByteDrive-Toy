"""离线 10 图层 BEV 栅格可视化模块的稳定公开入口。

模块: vis/bev_grid_vis/__init__.py
依赖: vis.bev_grid_vis.bev_grid_vis
读取配置: —（实现通过 cfg 读取）
对外接口:
    - BevGridReader(scene_dir) -> reader
    - render_bev_grid(grid, layer_names, cfg) -> ndarray
    - visualize_bev_grid(cfg, scene=None, frame=None, save=None, show=False) -> Path | None
"""

from vis.bev_grid_vis.bev_grid_vis import BevGridReader, render_bev_grid, visualize_bev_grid

__all__ = ["BevGridReader", "render_bev_grid", "visualize_bev_grid"]
