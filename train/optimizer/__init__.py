"""优化器构造：仅优化可训练参数（主干+三头），骨干冻结不纳入。公开 API 重导出入口。

模块: train/optimizer/__init__.py
依赖: train.optimizer.optimizer
读取配置: —（实现文件经 cfg 读取，本文件不读 config）
对外接口:
    - build_optimizer(model, cfg) -> AdamW   # 构造仅含可训练参数的优化器
说明: 跨模块统一 `from train.optimizer import ...`；实现见 optimizer.py，入参校验见 checks/。
"""

from train.optimizer.optimizer import build_optimizer

__all__ = ["build_optimizer"]
