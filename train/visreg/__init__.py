"""VISReg 损失模块的稳定公开入口。

模块: train/visreg/__init__.py
依赖: train.visreg.visreg
读取配置: —（实现通过 cfg 读取）
对外接口:
    - VISRegLoss(cfg) -> nn.Module
"""

from train.visreg.visreg import VISRegLoss

__all__ = ["VISRegLoss"]
