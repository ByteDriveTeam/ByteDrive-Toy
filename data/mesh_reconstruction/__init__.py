"""融合点云 Mesh 重建公开 API，支持可选水密修复。

模块: data/mesh_reconstruction/__init__.py
依赖: data.mesh_reconstruction.mesh_reconstruction
读取配置: —（配置由公开接口的 cfg 入参提供）
对外接口:
    - discover_pointclouds(input_path) -> list[Path]
    - reconstruct_scene(input_path, output_path, cfg) -> Path
    - run_reconstruction(cfg, input_path=None, output_dir=None, force=False) -> dict
"""

from data.mesh_reconstruction.mesh_reconstruction import (
    discover_pointclouds,
    reconstruct_scene,
    run_reconstruction,
)
from data.mesh_reconstruction.udf import reconstruct_udf_scene, run_udf_reconstruction

__all__ = ["discover_pointclouds", "reconstruct_scene", "run_reconstruction",
           "reconstruct_udf_scene", "run_udf_reconstruction"]
