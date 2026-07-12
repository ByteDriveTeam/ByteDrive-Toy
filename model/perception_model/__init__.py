"""多任务时序感知模型：冻结 DINOv3 骨干 + 3D 时序主干 + 语义/光流/深度三头。公开 API 重导出入口。

模块: model/perception_model/__init__.py
依赖: model.perception_model.perception_model
读取配置: —（实现文件经 cfg 读取，本文件不读 config）
对外接口:
    - PerceptionModel(cfg) -> nn.Module   # 多任务时序感知模型
说明: 跨模块统一 `from model.perception_model import ...`；实现见 perception_model.py，入参校验见 checks/。
"""

from model.perception_model.perception_model import PerceptionModel

__all__ = ["PerceptionModel"]
