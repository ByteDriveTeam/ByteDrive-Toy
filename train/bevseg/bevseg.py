"""BEVSeg 重建损失和独立训练循环。

模块: train/bevseg/bevseg.py
依赖: torch, train.bevseg.checks.bevseg_checks
读取配置: train.bevseg_loss_weights, train.grad_accum_steps/log_every/grad_clip_norm
对外接口:
    - compute_bevseg_losses(outputs, batch, cfg) -> (Tensor, dict)
    - train_bevseg_epoch(model, loader, optimizer, cfg, device) -> dict
    - evaluate_bevseg(model, loader, cfg, device) -> dict
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from train.bevseg.checks.bevseg_checks import check_bevseg_batch


def compute_bevseg_losses(outputs, batch, cfg):
    """计算语义 BCE、车道方向和分层词表熵正则。"""
    check_bevseg_batch(batch)
    weights = cfg.train.bevseg_loss_weights
    logits = outputs["reconstruction_logits"].reshape_as(batch["bevseg"])
    semantic_logits = logits[:, :, :10]
    semantic_target = batch["semantic"]
    positive = 1.0 + (weights.positive_weight - 1.0) * semantic_target
    semantic = (F.binary_cross_entropy_with_logits(
        semantic_logits, semantic_target, reduction="none") * positive).mean()
    direction_pred = logits[:, :, 10:12]
    direction_target = batch["direction"]
    lane_mask = semantic_target[:, :, 1:5].amax(2, keepdim=True)
    direction = (F.smooth_l1_loss(direction_pred, direction_target, reduction="none") * lane_mask).sum()
    direction = direction / lane_mask.expand_as(direction_pred).sum().clamp_min(1.0)
    probability = outputs["probabilities"].clamp_min(1e-8)
    entropy = -(probability * probability.log()).sum(-1).mean()
    base_probability = outputs["base_probabilities"]
    # 尖锐度作用于每个独立子词表的完整 16-way 分布，而非抽出的 Top4 子集。
    sharpening = 1.0 - base_probability.pow(2).sum(-1).mean()
    # 汇总 batch 和空间位置，但保留 64 个子词表分别统计其 16-way 使用率。
    usage = outputs["base_probabilities"].mean(dim=(0, 1))
    usage_entropy = -(usage * usage.clamp_min(1e-8).log()).sum(-1).mean()
    usage_loss = -usage_entropy
    total = (weights.semantic * semantic + weights.direction * direction +
             weights.entropy * entropy + weights.sharpening * sharpening +
             weights.usage * usage_loss)
    return total, {"semantic": semantic, "direction": direction,
                   "entropy": entropy, "sharpening": sharpening,
                   "usage_entropy": usage_entropy, "usage": usage_loss,
                   "total": total}


def train_bevseg_epoch(model, loader, optimizer, cfg, device, epoch=0):
    """训练一个 BEVSeg epoch。"""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    sums, count = {}, 0
    steps = len(loader)
    accumulation = cfg.train.grad_accum_steps
    for step, batch in enumerate(loader):
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        outputs = model(batch["bevseg"], epoch=epoch, sample=True)
        total, components = compute_bevseg_losses(outputs, batch, cfg)
        window = min(accumulation, steps - (step // accumulation) * accumulation)
        (total / window).backward()
        if (step + 1) % accumulation == 0 or step + 1 == steps:
            if cfg.train.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.trainable_parameters(), cfg.train.grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for name, value in components.items():
            sums[name] = sums.get(name, 0.0) + float(value.detach()) * batch["bevseg"].shape[0]
        count += batch["bevseg"].shape[0]
    return {name: value / max(count, 1) for name, value in sums.items()}


@torch.no_grad()
def evaluate_bevseg(model, loader, cfg, device):
    """评估一个 BEVSeg epoch。"""
    model.eval()
    sums, count = {}, 0
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        _, components = compute_bevseg_losses(model(batch["bevseg"], epoch=None, sample=False), batch, cfg)
        for name, value in components.items():
            sums[name] = sums.get(name, 0.0) + float(value) * batch["bevseg"].shape[0]
        count += batch["bevseg"].shape[0]
    return {name: value / max(count, 1) for name, value in sums.items()}
