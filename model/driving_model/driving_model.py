"""DrivingModel 统一别名重定向入口。

模块: model/driving_model/driving_model.py
依赖: model.driving_model.unified_driving_model
读取配置: —
对外接口:
    - DrivingModel(cfg) -> nn.Module
    - UnifiedDrivingModel(cfg) -> nn.Module
说明: 重定向至四帧块因果视觉与局部二维加速度词表主模型。
"""

from model.driving_model.unified_driving_model import UnifiedDrivingModel

DrivingModel = UnifiedDrivingModel

__all__ = ["DrivingModel", "UnifiedDrivingModel"]
