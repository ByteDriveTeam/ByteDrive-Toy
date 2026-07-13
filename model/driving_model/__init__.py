"""单帧开环驾驶模型：复用感知主干 → BEV → 三场 + 多模态轨迹。公开 API 重导出入口。

模块: model/driving_model/__init__.py
依赖: model.driving_model.driving_model
读取配置: —（转由 DrivingModel 读取 config.model.driving 等键）
对外接口:
    - DrivingModel(cfg) -> nn.Module
说明: 跨模块统一 `from model.driving_model import DrivingModel`；实现见 driving_model.py，校验见 checks/。
"""

from model.driving_model.driving_model import DrivingModel

__all__ = ["DrivingModel"]
