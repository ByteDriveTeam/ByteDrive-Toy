"""感知与新驾驶任务损失：独立占用、二维场、逐层 Detect 和流匹配。

模块: train/losses/losses.py
依赖: torch, scipy, config.schema.Config, data.target_encoding.physics_decode,
      train.losses.checks.losses_checks
读取配置: model.physics.semantic_ignore_index/symlog_scale/depth_max_m,
          train.loss_weights.*, train.driving_loss_weights.*, model.driving.detection,
          model.driving.bev
对外接口:
    - compute_losses(outputs, targets, cfg) -> (Tensor, dict[str, Tensor])
    - compute_driving_losses(outputs, targets, cfg) -> (Tensor, dict[str, Tensor])
说明: Detect 每层独立匹配，只对匹配项的最近未来 Mode 回归；流速度仅监督有效航点。
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
_VECTOR_EPS = 1e-6  # 向量运算仅需规避零模长，不能沿用掩码计数下限
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
    """独立计算场景/Agent 占用、二维场、六层 Detect 与流匹配损失。"""
    check_driving_losses_io(outputs, targets)
    weights = cfg.train.driving_loss_weights
    components = {
        "scene_occ": _masked_bce(outputs["scene_occ_logits"],
                                 targets["scene_occ"].float(),
                                 targets["scene_occ_mask"].float()),
        "agent_occ": _masked_bce(outputs["agent_occ_logits"],
                                 targets["agent_occ"].float(),
                                 targets["agent_occ_mask"].float()),
        "drivable": F.binary_cross_entropy_with_logits(
            outputs["drivable"][:, 0], targets["drivable"]),
        "lane_occupancy": F.binary_cross_entropy_with_logits(
            outputs["lane_occupancy"][:, 0], targets["lane_occupancy"]),
        "stop_line": _balanced_binary_loss(
            outputs["stop_line_logits"][:, 0], targets["stop_line"],
            torch.ones_like(targets["stop_line"])),
    }
    class_loss, box_loss, future_loss = _detect_losses(outputs, targets, cfg)
    components.update({"detect_class": class_loss, "detect_box": box_loss,
                       "detect_future": future_loss})
    valid = outputs["flow_valid"].float()
    flow_per = (outputs["flow_velocity"] - outputs["flow_target"]).square().sum(-1)
    components["flow"] = (flow_per * valid).sum() / valid.sum().clamp_min(1)
    total = sum(getattr(weights, name) * value for name, value in components.items())
    components["total"] = total
    return total, components


def _detect_losses(outputs, targets, cfg):
    """每层独立匈牙利匹配；未来仅监督最近的一个 Mode。"""
    from scipy.optimize import linear_sum_assignment

    logits = outputs["detect_class_logits"]
    boxes = outputs["detect_boxes"]
    futures = outputs["detect_future"]
    target_class = targets["detect_class"]
    target_box = targets["detect_box"]
    target_future = targets["detect_future"]
    target_valid = targets["detect_future_valid"]
    num_layers, batch, queries = logits.shape[:3]
    no_object = cfg.model.driving.detection.num_classes
    class_weight = logits.new_ones(no_object + 1)
    class_weight[-1] = cfg.model.driving.detection.no_object_weight
    bev = cfg.model.driving.bev
    det = cfg.model.driving.detection
    scale = logits.new_tensor((bev.x_max_m - bev.x_min_m, bev.y_max_m - bev.y_min_m,
                               bev.z_max_m - bev.z_min_m, *det.matching_size_scales_m,
                               torch.pi, det.matching_velocity_scale_mps,
                               det.matching_velocity_scale_mps))
    classification, box_terms, future_terms = [], [], []
    for layer in range(num_layers):
        for sample in range(batch):
            gt_indices = torch.nonzero(target_class[sample] >= 0, as_tuple=True)[0]
            labels = torch.full((queries,), no_object, device=logits.device,
                                dtype=torch.long)
            if len(gt_indices):
                gt_labels = target_class[sample, gt_indices]
                probability = logits[layer, sample].softmax(-1)
                cls_cost = -probability[:, gt_labels]
                box_cost = torch.cdist(
                    boxes[layer, sample] / scale,
                    target_box[sample, gt_indices] / scale, p=1)
                cost = (cls_cost + box_cost).detach().cpu().numpy()
                row, col = linear_sum_assignment(cost)
                row = torch.as_tensor(row, device=logits.device)
                matched = gt_indices[torch.as_tensor(col, device=logits.device)]
                labels[row] = target_class[sample, matched]
                box_terms.append(F.smooth_l1_loss(
                    boxes[layer, sample, row] / scale,
                    target_box[sample, matched] / scale))
                modes = futures[layer, sample, row]
                future_gt = target_future[sample, matched]
                valid = target_valid[sample, matched].float()
                distance = torch.linalg.vector_norm(modes - future_gt[:, None], dim=-1)
                mode_cost = (distance * valid[:, None]).sum(-1) / valid.sum(-1)[:, None].clamp_min(1)
                chosen = mode_cost.argmin(-1)
                selected = modes[torch.arange(len(row), device=logits.device), chosen]
                if bool(valid.any()):
                    future_terms.append(((torch.linalg.vector_norm(selected - future_gt, dim=-1)
                                          * valid).sum() / valid.sum()))
            classification.append(F.cross_entropy(logits[layer, sample], labels,
                                                   weight=class_weight))
    zero = logits.sum() * 0
    return (torch.stack(classification).mean(),
            torch.stack(box_terms).mean() if box_terms else zero,
            torch.stack(future_terms).mean() if future_terms else zero)

def _masked_bce(pred_logit: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """视场掩码下的 BCE-with-logits：sum(bce·mask)/max(sum(mask), eps)。"""
    per = F.binary_cross_entropy_with_logits(pred_logit, target, reduction="none") * mask
    return per.sum() / mask.sum().clamp_min(_MASK_EPS)


def _balanced_binary_loss(logits, target, valid):
    """前景/背景分别归一后等权；无停止线样本只保留背景项，避免稀疏正样本被淹没。"""
    per = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    positive = (target > 0.5).to(logits.dtype) * valid
    negative = (target <= 0.5).to(logits.dtype) * valid
    positive_count = positive.sum()
    negative_loss = (per * negative).sum() / negative.sum().clamp_min(_MASK_EPS)
    positive_loss = (per * positive).sum() / positive_count.clamp_min(_MASK_EPS)
    has_positive = (positive_count > 0).to(logits.dtype)
    return negative_loss + 0.5 * has_positive * (positive_loss - negative_loss)
