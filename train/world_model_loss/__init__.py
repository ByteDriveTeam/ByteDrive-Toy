"""世界模型掩码补全损失的稳定公开入口。

模块: train/world_model_loss/__init__.py
依赖: train.world_model_loss.world_model_loss
读取配置: —（实现通过 cfg 读取）
对外接口:
    - WorldModelReconstructionLoss(cfg) -> nn.Module
"""

from train.world_model_loss.world_model_loss import WorldModelReconstructionLoss

__all__ = ["WorldModelReconstructionLoss"]
