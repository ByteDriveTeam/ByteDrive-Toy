"""动态对象 Poisson、donor 复用与 Box 回退公开 API。

模块: data/mesh_reconstruction/dynamic/__init__.py
依赖: data.mesh_reconstruction.dynamic.dynamic
读取配置: —（配置由公开接口的 cfg 入参提供）
对外接口:
    - reconstruct_dynamic_objects(objects, source_voxel_size, cfg, device,
                                  work_dir) -> tuple[dict,dict]
"""

from data.mesh_reconstruction.dynamic.dynamic import reconstruct_dynamic_objects

__all__ = ["reconstruct_dynamic_objects"]
