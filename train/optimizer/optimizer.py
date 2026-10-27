"""优化器构造：仅优化任务前向实际使用的可训练参数，冻结或未参与前向的模块不纳入。

模块: train/optimizer/optimizer.py
依赖: torch, config.schema.Config, train.optimizer.checks.optimizer_checks
读取配置:
    train.lr
    train.weight_decay
    train.fused_optimizer
    train.perception_lr_scale（仅当模型提供 param_groups 时用于分组慢更新）
对外接口:
    - build_optimizer(model, cfg) -> torch.optim.AdamW
说明: 参数集合优先取 model.trainable_parameters()，普通 nn.Module 使用 requires_grad 参数。
      若模型提供 param_groups（如驾驶模型），则按分组构造：驾驶各件用 lr、感知子模块
      用 lr·perception_lr_scale 慢更新；驾驶模型不纳入未参与其前向的感知解码头。CUDA 按配置使用 fused
      AdamW，CPU/关闭配置时回退普通实现；标记 _no_weight_decay 的参数独立免衰减。各超参数来自 config。
"""

from __future__ import annotations

import torch

from config.schema import Config
from train.optimizer.checks.optimizer_checks import check_has_trainable


__all__ = ["build_optimizer"]


def build_optimizer(model, cfg: Config) -> torch.optim.AdamW:
    """构造 AdamW；模型提供 param_groups 时按分组（差分 lr）构造，否则单组优化全部可训练参数。"""
    if hasattr(model, "param_groups"):
        groups = model.param_groups(cfg.train.lr, cfg.train.weight_decay, cfg.train.perception_lr_scale)
    else:
        params = (model.trainable_parameters() if hasattr(model, "trainable_parameters")
                  else (p for p in model.parameters() if p.requires_grad))
        groups = [{"params": list(params), "lr": cfg.train.lr, "weight_decay": cfg.train.weight_decay}]
    groups = _split_weight_decay(groups, cfg.train.weight_decay)
    check_has_trainable([p for g in groups for p in g["params"]])
    return torch.optim.AdamW(groups, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
                             **_adamw_execution_kwargs(groups, cfg))


def _split_weight_decay(groups, default_decay):
    result = []
    for group in groups:
        params = list(group["params"])
        exempt = [p for p in params if getattr(p, "_no_weight_decay", False)]
        regular = [p for p in params if not getattr(p, "_no_weight_decay", False)]
        if regular:
            result.append({**group, "params": regular, "weight_decay": group.get("weight_decay", default_decay)})
        if exempt:
            result.append({**group, "params": exempt, "weight_decay": 0.0})
    return result


def _adamw_execution_kwargs(groups, cfg):
    """CUDA 按配置启用 fused AdamW；其余设备保持兼容实现。"""
    parameters = [parameter for group in groups for parameter in group["params"]]
    use_fused = bool(
        cfg.train.fused_optimizer
        and parameters
        and all(parameter.device.type == "cuda" for parameter in parameters))
    return {"fused": use_fused}
