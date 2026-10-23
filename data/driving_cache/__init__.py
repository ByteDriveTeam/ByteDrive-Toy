"""新 DrivingModel 训练切片缓存公共 API。

模块: data/driving_cache/__init__.py
依赖: data.driving_cache.driving_cache
读取配置: 由重导出实现读取
对外接口:
    - DrivingSliceCache(cfg) -> 缓存对象
    - DrivingSliceDataset(cfg) -> 数据集
    - navigation_command_from_route(route_geometry, route_index, cfg) -> int
    - pack_driving_sample(sample, words, residual_mean, residual_std, cfg) -> dict
    - prepare_driving_cache(samples, cfg) -> dict
说明: 其余重导出函数提供动作转换、标准化和切片选择。
"""

from data.driving_cache.driving_cache import (
    DrivingSliceCache,
    DrivingSliceDataset,
    trajectory_to_planar_accelerations,
    standardize_actions,
    unstandardize_actions,
    pack_driving_sample,
    navigation_command_from_route,
    slice_driving_samples,
    prepare_driving_cache,
)

__all__ = ["DrivingSliceCache", "DrivingSliceDataset", "trajectory_to_planar_accelerations",
           "standardize_actions", "unstandardize_actions", "pack_driving_sample",
           "navigation_command_from_route", "slice_driving_samples", "prepare_driving_cache"]
