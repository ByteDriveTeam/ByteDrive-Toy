"""双帧三目驾驶模型：融合三路几何图像与刚性对齐历史 BEV 的公开 API 入口。

模块: model/driving_model/__init__.py
依赖: model.driving_model.driving_model
读取配置: —（转由 DrivingModel 读取 config.model.driving 等键）
对外接口:
    - DrivingModel(cfg) -> nn.Module
说明: 跨模块统一 `from model.driving_model import DrivingModel`；实现见 driving_model.py，校验见 checks/。
"""

from model.driving_model.driving_model import DrivingModel

__all__ = ["DrivingModel"]
