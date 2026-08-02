"""静态世界与动态局部稀疏 TUDF 重建公开 API。

模块: data/mesh_reconstruction/udf/__init__.py
依赖: data.mesh_reconstruction.udf.udf
读取配置: —（配置由公开接口参数提供）
对外接口:
    - build_sparse_udf(points, tags, cfg, device, batch_size, candidate_budget,
                       max_voxels) -> dict
    - reconstruct_udf_scene(input_path, output_path, cfg) -> Path
    - run_udf_reconstruction(cfg, input_path=None, output_dir=None, force=False) -> dict
"""

from data.mesh_reconstruction.udf.udf import (
    build_sparse_udf,
    reconstruct_udf_scene,
    run_udf_reconstruction,
)

__all__ = ["build_sparse_udf", "reconstruct_udf_scene", "run_udf_reconstruction"]
