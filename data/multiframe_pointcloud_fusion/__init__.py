"""多帧语义 LiDAR 融合与动态对象重建公开 API。

模块: data/multiframe_pointcloud_fusion/__init__.py
依赖: data.multiframe_pointcloud_fusion.multiframe_pointcloud_fusion
读取配置: —（配置由公开接口的 cfg 入参提供）
对外接口:
    - discover_scenes(input_path) -> list[Path]
    - fuse_scene(scene_dir, output_dir, cfg) -> Path
    - run_fusion(cfg, input_path=None, output_dir=None) -> dict
"""

from data.multiframe_pointcloud_fusion.multiframe_pointcloud_fusion import (
    discover_scenes,
    fuse_scene,
    run_fusion,
)

__all__ = ["discover_scenes", "fuse_scene", "run_fusion"]

