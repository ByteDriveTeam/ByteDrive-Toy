"""多任务监督损失：感知多任务与驾驶三场、轨迹、置信度及 HDMap 越界约束。公开 API 重导出入口。

模块: train/losses/__init__.py
依赖: train.losses.losses
读取配置: —（实现文件经 cfg 读取，本文件不读 config）
对外接口:
    - compute_losses(outputs, targets, cfg) -> (Tensor, dict)          # 感知多任务加权损失
    - compute_driving_losses(outputs, targets, cfg) -> (Tensor, dict)  # 驾驶多任务加权损失
说明: 跨模块统一 `from train.losses import ...`；实现见 losses.py，入参校验见 checks/。
"""

from train.losses.losses import compute_driving_losses, compute_losses

__all__ = ["compute_losses", "compute_driving_losses"]
