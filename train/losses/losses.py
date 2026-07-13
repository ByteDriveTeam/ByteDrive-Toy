"""多任务监督损失：语义 CE + 深度 SmoothL1(掩码+距离加权) + 深度梯度 SmoothL1(掩码) + 深度范围 BCE。

模块: train/losses/losses.py
依赖: torch, config.schema.Config, data.target_encoding.physics_decode, train.losses.checks.losses_checks
读取配置:
    model.physics.semantic_ignore_index / symlog_scale / depth_max_m
    train.loss_weights.semantic / depth / depth_grad / depth_range
对外接口:
    - compute_losses(outputs, targets, cfg) -> (Tensor, dict[str, Tensor])   # 总损失与各分量（供日志）
说明: 单帧模型，outputs 为模型双头输出（[B,C,H,W]，FP32），targets 为数据集监督目标（batch 后 [B,H,W] 等）。
      深度 ch0 回归 scale·symlog(depth)、仅在范围内像素计损；ch1 以 BCE 监督范围内/超范围二分类（全像素）。
      语义 CE 忽略 Unlabeled。深度回归再叠「距离加权」：按 GT 深度(米)线性从近处 1 递减到远处 _DIST_WEIGHT_MIN，
      近距误差权重更高；权重并入掩码（分子分母同乘）得加权均值，不改变整体损失量级。范围二分类不加权
      （其目标即判定量程内外）。深度梯度：对 ch0 与 GT 的 H/W 相邻像素差取 SmoothL1，监督边界/结构清晰；
      仅用范围掩码、不加距离权（结构近远同等重要）。掩码归一用「有效（加权）像素数」而非全像素，避免超范围
      占比波动改变有效学习率。
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from config.schema import Config
from data.target_encoding import physics_decode
from train.losses.checks.losses_checks import check_losses_io


__all__ = ["compute_losses"]

_MASK_EPS = 1.0  # 掩码归一分母下限：一帧全超范围时避免除零，且不放大极少数有效像素的损失
_DIST_WEIGHT_MIN = 0.1  # 距离加权下限：depth=depth_max_m 处的权重，近处线性升至 1


def compute_losses(outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor],
                   cfg: Config) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """计算多任务加权总损失与各分量。"""
    check_losses_io(outputs, targets)
    weights = cfg.train.loss_weights
    physics = cfg.model.physics
    inrange = targets["depth_inrange"]  # [B,H,W]，1=范围内

    # 距离加权：解回 GT 深度(米)，按 depth_max_m 线性从近处 1 递减到远处 _DIST_WEIGHT_MIN，突出近距监督。
    # 权重乘进范围掩码，回归即对「范围内且按距离加权」的像素取加权均值（分子分母同乘，量级不变）。
    depth_m = physics_decode(targets["depth_target"], physics.symlog_scale)
    weighted_mask = inrange * _distance_weight(depth_m, physics.depth_max_m)  # [B,H,W]

    # 语义：logits [B,C,H,W] 与 long 标签 [B,H,W]，cross_entropy 原生支持多维
    semantic = F.cross_entropy(
        outputs["semantic"], targets["semantic"],
        ignore_index=physics.semantic_ignore_index)

    # 深度回归：ch0 对 scale·symlog(depth)，仅范围内像素、按距离加权
    depth_pred = outputs["depth"][:, 0]  # [B,H,W]
    depth = _masked_smooth_l1(depth_pred, targets["depth_target"], weighted_mask)
    # 深度梯度：ch0 与 GT 的 H/W 相邻像素差取 SmoothL1，监督边界/结构；仅用范围掩码、不加距离权
    depth_grad = _masked_gradient_l1(depth_pred, targets["depth_target"], inrange)
    # 深度范围二分类：ch1 logit 对 in_range，全像素 BCE（不加距离权：目标即判定量程内外）
    depth_range = F.binary_cross_entropy_with_logits(outputs["depth"][:, 1], inrange)

    total = (weights.semantic * semantic + weights.depth * depth + weights.depth_grad * depth_grad
             + weights.depth_range * depth_range)
    components = {"semantic": semantic, "depth": depth, "depth_grad": depth_grad,
                  "depth_range": depth_range, "total": total}
    return total, components


def _distance_weight(depth_m: torch.Tensor, depth_max_m: float) -> torch.Tensor:
    """按 GT 深度线性递减的距离权重：depth=0→1、depth=depth_max_m→_DIST_WEIGHT_MIN，钳到 [MIN, 1]。

    近处误差权重高、远处低（远处像素多、深度不确定性大）。超范围像素虽会被范围掩码乘零，仍钳下限防越界。
    """
    weight = 1.0 - (1.0 - _DIST_WEIGHT_MIN) * (depth_m / depth_max_m)
    return weight.clamp(_DIST_WEIGHT_MIN, 1.0)


def _masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """按 mask 归一的 SmoothL1：sum(loss·mask)/max(sum(mask), eps)。

    mask 先广播到 pred 形状再作分母，使多通道（光流 2 通道）分母与分子元素数一致、得逐元素均值。
    """
    mask = mask.expand_as(pred)
    per_element = F.smooth_l1_loss(pred, target, reduction="none") * mask
    return per_element.sum() / mask.sum().clamp_min(_MASK_EPS)


def _masked_gradient_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """空间梯度 SmoothL1：对 pred 与 target 的 H/W 相邻像素差取 SmoothL1，按有效边数归一。

    梯度定义在相邻像素对上，仅当两端像素都在范围内该差才有效，故边掩码 = 相邻两像素掩码相乘。
    两方向的加权和 / 有效边数，得逐边均值；不加距离权（结构清晰度近远同等重要）。
    """
    num_h, den_h = _axis_gradient_terms(pred, target, mask, -2)  # H 方向（行间差）
    num_w, den_w = _axis_gradient_terms(pred, target, mask, -1)  # W 方向（列间差）
    return (num_h + num_w) / (den_h + den_w).clamp_min(_MASK_EPS)


def _axis_gradient_terms(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                         dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """沿 dim 的相邻差：返回 (SmoothL1·边掩码之和, 边掩码之和)。边掩码=相邻两像素掩码相乘。"""
    n = mask.size(dim)
    edge = mask.narrow(dim, 0, n - 1) * mask.narrow(dim, 1, n - 1)
    per_edge = F.smooth_l1_loss(pred.diff(dim=dim), target.diff(dim=dim), reduction="none") * edge
    return per_edge.sum(), edge.sum()
