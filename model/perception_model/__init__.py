"""共享视觉特征编码器与语义/深度双头感知模型的公开 API 重导出入口。

模块: model/perception_model/__init__.py
依赖: model.perception_model.perception_model
读取配置: —（实现文件经 cfg 读取，本文件不读 config）
对外接口:
    - PerceptionFeatureEncoder(cfg) -> nn.Module   # 不含像素头的共享视觉编码器
    - PerceptionModel(cfg) -> nn.Module   # 多任务时序感知模型
说明: 跨模块统一 `from model.perception_model import ...`；实现见 perception_model.py，入参校验见 checks/。
"""

from model.perception_model.perception_model import PerceptionFeatureEncoder, PerceptionModel

__all__ = ["PerceptionFeatureEncoder", "PerceptionModel"]
