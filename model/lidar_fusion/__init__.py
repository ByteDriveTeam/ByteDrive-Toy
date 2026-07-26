"""LiDAR 体素特征与初始 BEV 查询融合的公开 API 重导出入口。

模块: model/lidar_fusion/__init__.py
依赖: model.lidar_fusion.lidar_fusion
读取配置: —（由 LidarQueryFusion 读取调用方传入的 config.model.driving）
对外接口:
    - LidarQueryFusion(cfg_driving) -> nn.Module
"""

from model.lidar_fusion.lidar_fusion import LidarQueryFusion

__all__ = ["LidarQueryFusion"]
