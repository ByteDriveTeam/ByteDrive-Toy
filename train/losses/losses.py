"""多任务监督损失：感知、驾驶场、道路线、交通控制、轨迹行为及安全约束。

模块: train/losses/losses.py
依赖: torch, config.schema.Config, data.target_encoding.(physics_decode, physics_target),
      train.losses.checks.losses_checks
读取配置:
    model.physics.semantic_ignore_index / symlog_scale / depth_max_m
    train.loss_weights.semantic / depth / depth_grad / depth_range
    train.driving_loss_weights.trajectory / confidence / behavior / distribution / risk / drivable /
        lane_class / lane_class_weights / lane_direction / boundary /
        stop_line / traffic_light_state / stop_crossing / trajectory_unmatched_weight
    model.driving.trajectory.symlog_scale
    model.driving.bev.x_min_m / x_max_m / y_min_m / y_max_m
    data.driving.traffic_control.stop_margin_m
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
from data.target_encoding import physics_decode, physics_target
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

    风险/可行驶为二值场：可行驶 GT 已从 HDMap 道路中扣除深度确认可见的 box 占用；二者在视场内掩码下做
    BCE（逼近 GT）。轨迹分布场性质不同——它是「能量/
    分数场」：目标是让 GT 航点处分数尽可能高，而非逼近某个固定值，故对视场内做空间 softmax 后与 GT 高斯软
    占据（归一化为分布）取交叉熵（只相对抬高 GT 邻域、压低其余）。轨迹按米制 ADE 的 8×1 匈牙利代价
    选择最相似 Mode，在 Symlog 空间全权重回归，其余 Mode 仍小权重更新；置信度学习匹配结果。行为固定八类彼此独立，以
    BCE-with-logits 监督同一帧同时激活的多个类别；样本始终保留在 batch 归一分母中。独立道路线图以加权
    CE 监督类别，并仅在道路线像素上以有符号余弦距离监督单位
    切向量；该方向来自 HD Map yaw，保留真实行驶正反向。越界损失把全部候选轨迹航点投影到不可行驶单侧距离场：道路外或可见 box 占用内
    按米制距离惩罚；超出 BEV 覆盖范围时另加坐标越界距离，保证仍有指向有效区域的梯度。
    """
    check_driving_losses_io(outputs, targets)
    w = cfg.train.driving_loss_weights
    inview = targets["inview"]  # [B,Hf,Wf]

    risk = _masked_bce(outputs["risk"][:, 0], targets["risk"], inview)
    drivable = _masked_bce(outputs["drivable"][:, 0], targets["drivable"], inview)
    distribution = _distribution_energy(outputs["distribution"][:, 0], targets["distribution"], inview)
    lane_weights = outputs["lane_class_logits"].new_tensor(w.lane_class_weights)
    lane_class = _masked_lane_ce(
        outputs["lane_class_logits"], targets["lane_class"], inview, lane_weights)
    lane_direction = _lane_direction_loss(
        outputs["lane_direction"], targets["lane_direction"], targets["lane_class"], inview)
    stop_line, traffic_light_state, stop_crossing = _traffic_control_losses(
        outputs, targets, cfg, inview)
    trajectory, confidence = _trajectory_losses(
        outputs["trajectory_normalized"], outputs["trajectories"], outputs["confidence"],
        targets["trajectory"], targets["traj_valid"],
        cfg.model.driving.trajectory.symlog_scale, w.trajectory_unmatched_weight)
    behavior = F.binary_cross_entropy_with_logits(outputs["behavior_logits"], targets["behavior"])
    boundary = _boundary_loss(
        outputs["trajectories"], targets["offroad_distance"], cfg.model.driving.bev)
    components = {"risk": risk, "drivable": drivable, "distribution": distribution,
                  "lane_class": lane_class, "lane_direction": lane_direction,
                  "stop_line": stop_line, "traffic_light_state": traffic_light_state,
                  "trajectory": trajectory, "confidence": confidence, "behavior": behavior,
                  "boundary": boundary, "stop_crossing": stop_crossing}
    total = sum(getattr(w, name) * loss for name, loss in components.items())
    components["total"] = total
    return total, components


def _masked_bce(pred_logit: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """视场掩码下的 BCE-with-logits：sum(bce·mask)/max(sum(mask), eps)。"""
    per = F.binary_cross_entropy_with_logits(pred_logit, target, reduction="none") * mask
    return per.sum() / mask.sum().clamp_min(_MASK_EPS)


def _masked_lane_ce(logits, target, inview, class_weights):
    """视场内道路线类别加权交叉熵；背景降权以免细线监督被数量淹没。"""
    per = F.cross_entropy(logits, target.long(), weight=class_weights, reduction="none") * inview
    return per.sum() / inview.sum().clamp_min(_MASK_EPS)


def _balanced_binary_loss(logits, target, valid):
    """前景/背景分别归一后等权；无停止线样本只保留背景项，避免稀疏正样本被淹没。"""
    per = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    positive = (target > 0.5).to(logits.dtype) * valid
    negative = (target <= 0.5).to(logits.dtype) * valid
    positive_count = positive.sum()
    negative_loss = (per * negative).sum() / negative.sum().clamp_min(_MASK_EPS)
    if not bool(positive_count > 0):
        return negative_loss
    positive_loss = (per * positive).sum() / positive_count
    return 0.5 * (positive_loss + negative_loss)


def _masked_state_ce(logits, target, valid):
    """仅在相关停止线且灯色已知的像素监督动态状态。"""
    per = F.cross_entropy(logits, target.long(), reduction="none") * valid
    return per.sum() / valid.sum().clamp_min(_MASK_EPS)


def _traffic_control_losses(outputs, targets, cfg, inview):
    """集中计算停止线、灯态与红灯越线三项相关损失。"""
    stop_line = _balanced_binary_loss(
        outputs["stop_line_logits"][:, 0], targets["stop_line"], inview)
    traffic_state = _masked_state_ce(
        outputs["traffic_light_state_logits"], targets["traffic_light_state"],
        targets["traffic_light_state_valid"] * inview)
    stop_crossing = _stop_crossing_loss(
        outputs["trajectories"], targets["stop_point"], targets["stop_direction"],
        targets["red_stop_valid"], cfg.data.driving.traffic_control.stop_margin_m)
    return stop_line, traffic_state, stop_crossing


def _lane_direction_loss(pred, target, lane_class, inview):
    """仅道路线像素上的有向余弦距离；`v` 与 `-v` 不等价，保留真实行驶方向。"""
    pred_unit = F.normalize(pred, dim=1, eps=_MASK_EPS)
    target_unit = F.normalize(target, dim=1, eps=_MASK_EPS)
    direction_valid = target.square().sum(1) > _MASK_EPS
    mask = ((lane_class > 0) & direction_valid).to(pred.dtype) * inview
    per = (1.0 - (pred_unit * target_unit).sum(1)) * mask
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


def _trajectory_losses(trajectories_normalized: torch.Tensor, trajectories_meters: torch.Tensor,
                       confidence: torch.Tensor, gt_meters: torch.Tensor, valid: torch.Tensor,
                       symlog_scale: float, unmatched_weight: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """8×1 匈牙利匹配后的 Symlog 轨迹监督，未匹配 Mode 仍小权重更新。

    每个样本只有一条专家 GT，米制平均位移代价上的矩形匈牙利精确解就是代价最小的 Mode。匹配 Mode
    权重为 1，其余 Mode 使用 `unmatched_weight`；全部样本仍保留在 batch 均值分母中。缺少未来航点的样本贡献可微零值，
    不会被数据管线或 batch 过滤掉。
    """
    waypoint_mask = valid[:, None]                               # [B,1,T]
    displacement = torch.linalg.vector_norm(trajectories_meters - gt_meters[:, None], dim=-1)
    valid_count = valid.sum(-1)                                  # [B]
    matching_cost = (displacement * waypoint_mask).sum(-1) \
        / valid_count[:, None].clamp_min(_MASK_EPS)
    matched_modes = _hungarian_single_target(matching_cost)

    gt_normalized = physics_target(gt_meters, symlog_scale)
    expanded_gt = gt_normalized[:, None].expand_as(trajectories_normalized)
    per_waypoint = F.smooth_l1_loss(
        trajectories_normalized, expanded_gt, reduction="none").sum(-1)
    per_waypoint = per_waypoint * waypoint_mask                  # [B,M,T]
    mode_weights = torch.full_like(matching_cost, unmatched_weight)
    mode_weights.scatter_(1, matched_modes[:, None], 1.0)
    numerator = (per_waypoint * mode_weights[:, :, None]).sum((1, 2))
    denominator = valid_count * mode_weights.sum(1)
    sample_valid = (valid_count > 0).to(numerator.dtype)
    trajectory = (numerator / denominator.clamp_min(_MASK_EPS) * sample_valid).mean()
    confidence_loss = F.cross_entropy(confidence, matched_modes, reduction="none")
    confidence_loss = (confidence_loss * sample_valid).mean()
    return trajectory, confidence_loss


def _hungarian_single_target(cost: torch.Tensor) -> torch.Tensor:
    """求每个 `[M,1]` 代价矩阵的匈牙利匹配；单 GT 时精确退化为逐 Mode 最小值索引。"""
    return cost.detach().argmin(dim=1)


def _boundary_loss(trajectories: torch.Tensor, offroad_distance: torch.Tensor, bev) -> torch.Tensor:
    """轨迹越界损失：可微采样道路外/可见占用距离场，并惩罚超出 BEV 覆盖范围的部分。

    对全部模态与全部航点等权约束，避免低置信度候选轨迹逃逸到道路外或穿过占用。`grid_sample` 的最后一维依次是列、行，
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


def _stop_crossing_loss(trajectories, stop_point, stop_direction, red_valid, stop_margin_m):
    """红灯时按路线切向惩罚越过安全停止位置的全部候选航点。

    `dot(point-stop_point, direction)` 在停止线之后为正；加上安全余量后，允许区域截止于停止线前
    `stop_margin_m`。所有模态都受约束，避免低置信度越线轨迹在闭环选择时成为安全漏洞。
    """
    relative = trajectories - stop_point[:, None, None, :]
    signed = (relative * stop_direction[:, None, None, :]).sum(-1) + stop_margin_m
    mask = red_valid[:, None, None].to(trajectories.dtype)
    count = mask.sum() * trajectories.shape[1] * trajectories.shape[2]
    return (F.relu(signed) * mask).sum() / count.clamp_min(_MASK_EPS)
