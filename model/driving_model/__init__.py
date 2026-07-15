"""双帧开环驾驶模型：当前图像后融合刚性对齐的上一帧 BEV，并解码道路线图与驾驶输出。公开 API 重导出入口。

模块: model/driving_model/__init__.py
依赖: model.driving_model.driving_model
读取配置: —（转由 DrivingModel 读取 config.model.driving 等键）
对外接口:
    - DrivingModel(cfg) -> nn.Module
说明: 跨模块统一 `from model.driving_model import DrivingModel`；实现见 driving_model.py，校验见 checks/。
"""

from model.driving_model.driving_model import DrivingModel

__all__ = ["DrivingModel"]
