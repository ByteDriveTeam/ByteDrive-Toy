"""新 DrivingModel 的二维加速度词选择、残差流匹配与训练循环。

模块: train/unified_driving/unified_driving.py
依赖: torch, config.schema.Config, model.driving_model.UnifiedDrivingModel
读取配置: train.precision/grad_accum_steps/grad_clip_norm/log_every
对外接口:
    - compute_flow_matching_losses(model, batch, cfg) -> (Tensor, dict)
    - train_unified_driving_epoch(model, loader, optimizer, cfg, device) -> dict
说明: 残差噪声插值和终点监督处在二维加速度残差的 Z-Score 空间。
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["compute_flow_matching_losses", "train_unified_driving_epoch"]


def compute_flow_matching_losses(model, batch: Dict[str, torch.Tensor], cfg):
    """用缓存的 Top1 词及其标准化物理残差计算联合损失。"""
    device = batch["rgb"].device
    batch_size = int(batch["rgb"].shape[0])
    time_step = torch.rand(batch_size, device=device)
    residual_target = batch["trajectory_residual"].float()
    decoder = model.trajectory_decoder
    nearest = batch["trajectory_word_index"].long()
    action_noise = torch.randn_like(residual_target)
    action_state = (1.0 - time_step[:, None, None]) * action_noise \
        + time_step[:, None, None] * residual_target
    if "bev_latent" not in batch and "bev_grid" in batch:
        batch = dict(batch)
        batch["bev_latent"] = model.encode_bev_targets(batch["bev_grid"])
    bev_state = None
    bev_velocity = None
    if "bev_latent" in batch:
        bev_target = batch["bev_latent"].float().reshape(
            batch_size, -1, batch["bev_latent"].shape[-2], batch["bev_latent"].shape[-1])
        bev_noise = torch.randn_like(bev_target)
        bev_state = (1.0 - time_step[:, None, None, None]) * bev_noise \
            + time_step[:, None, None, None] * bev_target
        bev_velocity = bev_target - bev_noise
    outputs = model(
        batch["rgb"], batch["intrinsics"], batch["navigation_command"], batch["ego_state"],
        batch["motion_matrices"], time_step=time_step, traj_state=action_state,
        traj_anchor_index=nearest, bev_state=bev_state)
    components = {}
    # 终点 MSE 等价于对端点参数化条件流匹配施加 (1-t)^2 权重。
    components["trajectory"] = (outputs["traj_residual"] - residual_target).square().mean()
    targets = F.one_hot(nearest, decoder.actions.shape[0]).float()
    if not all(name in batch for name in ("word_risks", "gt_risks")):
        raise KeyError("新版驾驶缓存必须提供 word_risks 和 gt_risks")
    extra_risk = (batch["word_risks"].bool()
                  & ~batch["gt_risks"].bool()[:, None, :]).any(dim=-1)
    weight = torch.where(extra_risk & (targets == 0),
                         torch.full_like(targets, cfg.model.unified_driving.risk_negative_weight),
                         torch.ones_like(targets))
    components["selection"] = (F.binary_cross_entropy_with_logits(
        outputs["traj_logits"].float(), targets, reduction="none") * weight).mean()
    annotation_losses = []
    if "annotation_valid" in batch:
        annotation_losses.append(_annotation_loss(outputs["visual_annotation"], batch))
    components["annotation"] = (sum(annotation_losses) / len(annotation_losses)
                                 if annotation_losses else residual_target.new_zeros(()))
    if "bev_latent" in batch:
        components["bev"] = (outputs["bev_flow"].float() - bev_velocity).square().mean()
    else:
        components["bev"] = residual_target.new_zeros(())
    total = components["trajectory"] + components["selection"] + components["bev"] \
        + cfg.model.unified_driving.annotation_loss_weight * components["annotation"]
    components["total"] = total
    return total, components


def _annotation_loss(prediction, batch):
    """只监督有定义的驾驶状态、决策原因、冲突概率与车道可用性。"""
    valid = batch["annotation_valid"].float().view(-1)
    state_target = batch["annotation_state"].long().view(-1)
    reason_target = batch["annotation_reason"].long().view(-1)
    state = prediction["state_logits"].float()
    reason = prediction["reason_logits"].float()
    state_loss = F.cross_entropy(state, state_target, ignore_index=-1, reduction="none")
    reason_loss = F.cross_entropy(reason, reason_target, ignore_index=-1, reduction="none")
    state_mask = valid * (state_target >= 0).float()
    reason_mask = valid * (reason_target >= 0).float()
    probability_mask = valid[:, None] * batch["annotation_probability_valid"].float()
    probability_loss = F.binary_cross_entropy_with_logits(
        prediction["probability_logits"].float(),
        batch["annotation_probabilities"].float(), reduction="none")
    return (state_loss * state_mask).sum() / state_mask.sum().clamp_min(1.0) \
        + (reason_loss * reason_mask).sum() / reason_mask.sum().clamp_min(1.0) \
        + (probability_loss * probability_mask).sum() / probability_mask.sum().clamp_min(1.0)


def train_unified_driving_epoch(model, loader, optimizer, cfg, device):
    """训练新模型一个 epoch。"""
    model.train()
    meter = _Meter()
    optimizer.zero_grad(set_to_none=True)
    steps = len(loader)
    accumulation = cfg.train.grad_accum_steps
    for step, batch in enumerate(loader):
        batch = {name: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                 for name, value in batch.items()}
        total, components = compute_flow_matching_losses(model, batch, cfg)
        window = min(accumulation, steps - (step // accumulation) * accumulation)
        (total / window).backward()
        if (step + 1) % accumulation == 0 or step + 1 == steps:
            if cfg.train.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.trainable_parameters(), cfg.train.grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        meter.update(components, int(batch["rgb"].shape[0]))
        if cfg.train.log_every > 0 and step % cfg.train.log_every == 0:
            print("[unified-driving] step {} {}".format(step, _format(components)))
    return meter.averages()


class _Meter:
    """按样本聚合损失日志。"""

    def __init__(self):
        self.values, self.count = {}, 0

    def update(self, values, count):
        self.count += count
        for name, value in values.items():
            self.values[name] = self.values.get(name, 0.0) + float(value.detach()) * count

    def averages(self):
        return {name: value / max(self.count, 1) for name, value in self.values.items()}


def _format(values):
    return "  ".join("{}={:.4f}".format(name, value.detach().item())
                     for name, value in values.items())
