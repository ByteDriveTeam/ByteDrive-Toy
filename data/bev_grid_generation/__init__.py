"""离线 BEV 特权栅格生成模块的稳定公开入口。

模块: data/bev_grid_generation/__init__.py
依赖: data.bev_grid_generation.bev_grid_generation
读取配置: —（实现通过 cfg 读取，入口只重导出）
对外接口:
    - generate_bev_grids(cfg, scene_limit=None) -> dict
    - read_grid_scene_meta(scene_dir) -> dict
"""

from data.bev_grid_generation.bev_grid_generation import generate_bev_grids, read_grid_scene_meta

__all__ = ["generate_bev_grids", "read_grid_scene_meta"]
