"""优化器构造：仅优化可训练参数（主干+三头），骨干冻结不纳入。

模块: train/optimizer/optimizer.py
依赖: torch, config.schema.Config, train.optimizer.checks.optimizer_checks
读取配置:
    train.lr
    train.weight_decay
对外接口:
    - build_optimizer(model, cfg) -> torch.optim.AdamW
说明: 参数集合取 model.trainable_parameters()（排除冻结的 DINOv3 骨干），避免把 requires_grad=False
      的骨干参数交给优化器。lr / weight_decay 唯一来源为 config（规范 §6）。
"""

from __future__ import annotations

import torch

from config.schema import Config
from train.optimizer.checks.optimizer_checks import check_has_trainable


__all__ = ["build_optimizer"]


def build_optimizer(model, cfg: Config) -> torch.optim.AdamW:
    """构造 AdamW，仅优化 model 的可训练参数。"""
    params = list(model.trainable_parameters())
    check_has_trainable(params)
    return torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
