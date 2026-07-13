"""训练与评估循环：前向 → 多任务损失 → 反向 → 梯度裁剪 → 步进，并聚合日志。公开 API 重导出入口。

模块: train/loop/__init__.py
依赖: train.loop.loop
读取配置: —（实现文件经 cfg 读取，本文件不读 config）
对外接口:
    - train_one_epoch(model, loader, optimizer, cfg, device) -> dict      # 感知训练一个 epoch
    - evaluate(...) -> dict                                               # 感知评估循环
    - train_driving_epoch(model, loader, optimizer, cfg, device) -> dict  # 驾驶训练一个 epoch
    - evaluate_driving(...) -> dict                                       # 驾驶评估循环
说明: 跨模块统一 `from train.loop import ...`；实现见 loop.py，入参校验见 checks/。
"""

from train.loop.loop import evaluate, evaluate_driving, train_driving_epoch, train_one_epoch

__all__ = ["train_one_epoch", "evaluate", "train_driving_epoch", "evaluate_driving"]
