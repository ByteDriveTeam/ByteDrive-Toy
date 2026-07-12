"""多任务监督损失：语义 CE + 深度 SmoothL1(掩码) + 深度范围 BCE + 光流 SmoothL1(掩码)。公开 API 重导出入口。

模块: train/losses/__init__.py
依赖: train.losses.losses
读取配置: —（实现文件经 cfg 读取，本文件不读 config）
对外接口:
    - compute_losses(...) -> dict   # 汇总四任务加权损失
说明: 跨模块统一 `from train.losses import ...`；实现见 losses.py，入参校验见 checks/。
"""

from train.losses.losses import compute_losses

__all__ = ["compute_losses"]
