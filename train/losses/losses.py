"""多任务监督损失：语义 CE + 深度 SmoothL1(掩码) + 深度范围 BCE + 光流 SmoothL1(掩码)。

模块: train/losses/losses.py
依赖: torch, config.schema.Config, train.losses.checks.losses_checks
读取配置:
    model.physics.semantic_ignore_index
    train.loss_weights.semantic / depth / depth_range / flow
对外接口:
    - compute_losses(outputs, targets, cfg) -> (Tensor, dict[str, Tensor])   # 总损失与各分量（供日志）
说明: outputs 为模型三头输出（[B,C,T,H,W]，FP32），targets 为数据集监督目标（batch 后 [B,T,H,W] 等）。
      深度 ch0 回归 scale·symlog(depth)、仅在范围内像素计损；ch1 以 BCE 监督范围内/超范围二分类（全像素）。
      光流回归 scale·symlog(速度)，同样掩到范围内（超范围如天空无稳定物理速度）。语义 CE 忽略 Unlabeled。
      掩码归一用「有效像素数」而非全像素，避免超范围占比波动改变有效学习率。
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from config.schema import Config
from train.losses.checks.losses_checks import check_losses_io


__all__ = ["compute_losses"]

_MASK_EPS = 1.0  # 掩码归一分母下限：一帧全超范围时避免除零，且不放大极少数有效像素的损失


def compute_losses(outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor],
                   cfg: Config) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """计算多任务加权总损失与各分量。"""
    check_losses_io(outputs, targets)
    weights = cfg.train.loss_weights
    inrange = targets["depth_inrange"]  # [B,T,H,W]，1=范围内

    # 语义：logits [B,C,T,H,W] 与 long 标签 [B,T,H,W]，cross_entropy 原生支持多维
    semantic = F.cross_entropy(
        outputs["semantic"], targets["semantic"],
        ignore_index=cfg.model.physics.semantic_ignore_index)

    # 深度回归：ch0 对 scale·symlog(depth)，仅范围内像素
    depth_pred = outputs["depth"][:, 0]  # [B,T,H,W]
    depth = _masked_smooth_l1(depth_pred, targets["depth_target"], inrange)
    # 深度范围二分类：ch1 logit 对 in_range，全像素 BCE
    depth_range = F.binary_cross_entropy_with_logits(outputs["depth"][:, 1], inrange)

    # 光流回归：[B,2,T,H,W] 对速度目标，掩到范围内（掩码在通道维广播）
    flow = _masked_smooth_l1(outputs["flow"], targets["flow_target"], inrange.unsqueeze(1))

    total = (weights.semantic * semantic + weights.depth * depth
             + weights.depth_range * depth_range + weights.flow * flow)
    components = {"semantic": semantic, "depth": depth, "depth_range": depth_range,
                  "flow": flow, "total": total}
    return total, components


def _masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """按 mask 归一的 SmoothL1：sum(loss·mask)/max(sum(mask), eps)。

    mask 先广播到 pred 形状再作分母，使多通道（光流 2 通道）分母与分子元素数一致、得逐元素均值。
    """
    mask = mask.expand_as(pred)
    per_element = F.smooth_l1_loss(pred, target, reduction="none") * mask
    return per_element.sum() / mask.sum().clamp_min(_MASK_EPS)
