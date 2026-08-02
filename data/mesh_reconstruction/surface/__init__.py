"""Poisson 表面重建与可选水密修复公开 API。

模块: data/mesh_reconstruction/surface/__init__.py
依赖: data.mesh_reconstruction.surface.surface
读取配置: —（配置由公开接口的 cfg 入参提供）
对外接口:
    - reconstruct_surface(points, tags, orientation_targets, cfg, hole_radius_m,
                          enable_watertight_repair, device, batch_size,
                          candidate_budget, poisson_threads) -> dict
    - reconstruct_surface_isolated(points, tags, orientation_targets, cfg, hole_radius_m,
                                   enable_watertight_repair, device, batch_size,
                                   candidate_budget, poisson_threads, work_dir) -> dict
"""

from data.mesh_reconstruction.surface.surface import (
    reconstruct_surface,
    reconstruct_surface_isolated,
)

__all__ = ["reconstruct_surface", "reconstruct_surface_isolated"]
