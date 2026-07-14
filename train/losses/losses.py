"""多任务监督损失：感知多任务与驾驶三场、轨迹、置信度及 HDMap 越界约束。

模块: train/losses/losses.py
依赖: torch, config.schema.Config, data.target_encoding.physics_decode, train.losses.checks.losses_checks
读取配置:
    model.physics.semantic_ignore_index / symlog_scale / depth_max_m
    train.loss_weights.semantic / depth / depth_grad / depth_range
    train.driving_loss_weights.trajectory / confidence / distribution / risk / drivable / boundary
    model.driving.bev.x_min_m / x_max_m / y_min_m / y_max_m
对外接口:
    - compute_losses(outputs, targets, cfg) -> (Tensor, dict[str, Tensor])          # 感知总损失与各分量
    - compute_driving_losses(outputs, targets, cfg) -> (Tensor, dict[str, Tensor])  # 驾驶总损失与各分量
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
from train.losses.checks.losses_checks import check_driving_losses_io, check_losses_io


__all__ = ["compute_losses", "compute_driving_losses"]

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


def compute_driving_losses(outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor],
                           cfg: Config) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """计算驾驶多任务加权总损失与各分量。

    风险/可行驶为占据场（{0,1} 硬标签），在视场内掩码下做 BCE（逼近 GT）。轨迹分布场性质不同——它是「能量/
    分数场」：目标是让 GT 航点处分数尽可能高，而非逼近某个固定值，故对视场内做空间 softmax 后与 GT 高斯软
    占据（归一化为分布）取交叉熵（只相对抬高 GT 邻域、压低其余）。轨迹为 winner-take-all 多模态：每样本 GT 按
    其扇区选中对应模态回归（SmoothL1，仅有效航点），并以该扇区为标签对置信度做交叉熵。视场外/无有效前向 GT
    的样本自动被掩码剔除。越界损失把全部候选轨迹航点投影到 HDMap 单侧距离场，道路内为零、道路外按米制
    距离惩罚；超出 BEV 覆盖范围时另加坐标越界距离，保证仍有指向有效区域的梯度。
    """
    check_driving_losses_io(outputs, targets)
    w = cfg.train.driving_loss_weights
    inview = targets["inview"]  # [B,Hf,Wf]

    risk = _masked_bce(outputs["risk"][:, 0], targets["risk"], inview)
    drivable = _masked_bce(outputs["drivable"][:, 0], targets["drivable"], inview)
    distribution = _distribution_energy(outputs["distribution"][:, 0], targets["distribution"], inview)
    trajectory, confidence = _trajectory_losses(
        outputs["trajectories"], outputs["confidence"],
        targets["trajectory"], targets["traj_valid"], targets["sector"])
    boundary = _boundary_loss(
        outputs["trajectories"], targets["offroad_distance"], cfg.model.driving.bev)

    total = (w.risk * risk + w.drivable * drivable + w.distribution * distribution
             + w.trajectory * trajectory + w.confidence * confidence + w.boundary * boundary)
    components = {"risk": risk, "drivable": drivable, "distribution": distribution,
                  "trajectory": trajectory, "confidence": confidence, "boundary": boundary,
                  "total": total}
    return total, components


def _masked_bce(pred_logit: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """视场掩码下的 BCE-with-logits：sum(bce·mask)/max(sum(mask), eps)。"""
    per = F.binary_cross_entropy_with_logits(pred_logit, target, reduction="none") * mask
    return per.sum() / mask.sum().clamp_min(_MASK_EPS)


def _distribution_energy(field_logit: torch.Tensor, target_soft: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:
    """轨迹分布能量损失：视场内空间 softmax，最大化 GT 软占据处的对数概率（GT 分数高）。

    把场当作 BEV 上的未归一化分数：仅在视场内竞争（softmax），与 GT 高斯软占据（视场内归一化为分布）做
    交叉熵。相比 BCE 逼近固定目标，本式只相对抬高 GT 邻域、压低其余，符合「预测一个场使 GT 分数尽可能高」。
    无 GT 的样本（视场内软占据全 0）贡献 0。
    """
    b = field_logit.shape[0]
    flat = field_logit.reshape(b, -1)
    in_mask = mask.reshape(b, -1) > 0
    logp = torch.log_softmax(flat.masked_fill(~in_mask, torch.finfo(flat.dtype).min), dim=1)
    tgt = target_soft.reshape(b, -1) * in_mask
    tgt = tgt / tgt.sum(1, keepdim=True).clamp_min(_MASK_EPS)     # 视场内归一化为概率分布
    # 仅在 tgt>0（视场内、有 GT 权重）处累计，避免 (-inf)·0 的 NaN
    contrib = torch.where(tgt > 0, tgt * logp, torch.zeros_like(logp))
    return (-contrib.sum(1)).mean()


def _trajectory_losses(trajectories: torch.Tensor, confidence: torch.Tensor, gt: torch.Tensor,
                       valid: torch.Tensor, sector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """WTA 多模态轨迹回归 + 扇区置信度分类。

    trajectories `[B,M,T,2]`、confidence `[B,M]`、gt `[B,T,2]`、valid `[B,T]`、sector `[B]`（-1=无效样本）。
    按 GT 所属扇区选中对应模态回归（SmoothL1，逐航点按 valid 掩码），以扇区为标签对置信度做 CE；无有效前向
    GT 的样本（sector<0）不计损。
    """
    b = trajectories.shape[0]
    sample_valid = sector >= 0                                   # [B]
    if not bool(sample_valid.any()):
        zero = trajectories.sum() * 0.0                          # 保持计算图/设备一致的 0
        return zero, zero

    picked = trajectories[torch.arange(b, device=trajectories.device), sector.clamp_min(0)]  # [B,T,2]
    wp_mask = valid * sample_valid[:, None].to(valid.dtype)      # [B,T]
    per_wp = F.smooth_l1_loss(picked, gt, reduction="none").sum(-1) * wp_mask  # [B,T]
    trajectory = per_wp.sum() / wp_mask.sum().clamp_min(_MASK_EPS)
    confidence = F.cross_entropy(confidence[sample_valid], sector[sample_valid])
    return trajectory, confidence


def _boundary_loss(trajectories: torch.Tensor, offroad_distance: torch.Tensor, bev) -> torch.Tensor:
    """HDMap 轨迹越界损失：可微采样道路外距离场，并惩罚超出 BEV 覆盖范围的部分。

    对全部模态与全部航点等权约束，避免低置信度候选轨迹逃逸到道路外。`grid_sample` 的最后一维依次是列、行，
    因而 ego 的 `(x前向, y右向)` 要换成 `(y归一列, x反向归一行)`。
    """
    x, y = trajectories[..., 0], trajectories[..., 1]
    grid_col = 2.0 * (y - bev.y_min_m) / (bev.y_max_m - bev.y_min_m) - 1.0
    grid_row = 1.0 - 2.0 * (x - bev.x_min_m) / (bev.x_max_m - bev.x_min_m)
    grid = torch.stack((grid_col, grid_row), dim=-1)               # [B,M,T,2]
    sampled = F.grid_sample(
        offroad_distance[:, None], grid, mode="bilinear", padding_mode="border",
        align_corners=False)[:, 0]                                 # [B,M,T]，单位米

    x_over = F.relu(bev.x_min_m - x) + F.relu(x - bev.x_max_m)
    y_over = F.relu(bev.y_min_m - y) + F.relu(y - bev.y_max_m)
    return (sampled + x_over + y_over).mean()
