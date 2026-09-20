"""BEVSeg 重建损失和独立训练循环。

模块: train/bevseg/bevseg.py
依赖: torch, train.bevseg.checks.bevseg_checks
读取配置: model.bevseg.stochastic_sampling, train.bevseg_loss_weights,
          train.bevseg_gradient_monitor, train.grad_accum_steps/log_every/grad_clip_norm
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
    semantic_logits = logits[:, :10]
    semantic_target = batch["semantic"]
    class_losses = []
    class_weights = []
    for index, name in enumerate(cfg.data.bevseg.layers):
        item = weights.semantic_classes[name]
        bce = F.binary_cross_entropy_with_logits(
            semantic_logits[:, index], semantic_target[:, index], reduction="none")
        pixel_weight = 1.0 + (item.positive_weight - 1.0) * semantic_target[:, index]
        class_loss = (bce * pixel_weight).sum() / pixel_weight.sum().clamp_min(1.0)
        class_losses.append(class_loss * item.bce_weight)
        class_weights.append(item.bce_weight)
    semantic = torch.stack(class_losses).sum() / torch.as_tensor(
        class_weights, device=semantic_logits.device, dtype=semantic_logits.dtype).sum().clamp_min(1e-8)
    direction_pred = logits[:, 10:12]
    direction_target = batch["direction"]
    lane_mask = semantic_target[:, 1:5].amax(1, keepdim=True)
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
    # 每个子词表统计当前 batch 全部 patch 的不同 Top1 码字数，再对 64 组取平均。
    top1 = base_probability.argmax(-1)
    top1_used = F.one_hot(top1, num_classes=base_probability.shape[-1]) \
        .flatten(0, 1).any(0).sum(-1).float().mean()
    total = (semantic + weights.direction * direction +
             weights.entropy * entropy + weights.sharpening * sharpening +
             weights.usage * usage_loss)
    return total, {"semantic": semantic, "direction": direction,
                   "entropy": entropy, "sharpening": sharpening,
                   "usage_entropy": usage_entropy, "usage": usage_loss,
                   "top1_usage_count": top1_used,
                   "total": total}


def train_bevseg_epoch(model, loader, optimizer, cfg, device, epoch=0):
    """训练一个 BEVSeg epoch，并按配置输出当前步损失与进度。"""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    sums, count = {}, 0
    steps = len(loader)
    accumulation = cfg.train.grad_accum_steps
    gradient_cfg = cfg.train.bevseg_gradient_monitor
    for step, batch in enumerate(loader):
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        outputs = model(batch["bevseg"], epoch=epoch,
                        sample=cfg.model.bevseg.stochastic_sampling)
        total, components = compute_bevseg_losses(outputs, batch, cfg)
        window = min(accumulation, steps - (step // accumulation) * accumulation)
        (total / window).backward()
        should_log = step == 0 or (step + 1) % cfg.train.log_every == 0 or step + 1 == steps
        gradient_metrics = (_gradient_metrics(model, gradient_cfg.small_abs_threshold)
                            if gradient_cfg.enabled and should_log else {})
        if (step + 1) % accumulation == 0 or step + 1 == steps:
            if cfg.train.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.trainable_parameters(), cfg.train.grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        batch_size = batch["bevseg"].shape[0]
        for name, value in components.items():
            contribution = value.detach() * batch_size
            sums[name] = sums[name] + contribution if name in sums else contribution
        count += batch_size
        if should_log:
            logged = {**components, **gradient_metrics}
            print("[bevseg] epoch {}/{} step {}/{} ({:.2f}%) {}".format(
                epoch + 1, cfg.train.epochs, step + 1, steps,
                (step + 1) * 100.0 / steps, _format_losses(logged)), flush=True)
    return _averages(sums, count)


@torch.no_grad()
def evaluate_bevseg(model, loader, cfg, device):
    """评估一个 BEVSeg epoch。"""
    model.eval()
    sums, count = {}, 0
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        _, components = compute_bevseg_losses(model(batch["bevseg"], epoch=None, sample=False), batch, cfg)
        batch_size = batch["bevseg"].shape[0]
        for name, value in components.items():
            contribution = value.detach() * batch_size
            sums[name] = sums[name] + contribution if name in sums else contribution
        count += batch_size
    return _averages(sums, count)


def _averages(sums, count):
    return {name: (value / max(count, 1)).item() for name, value in sums.items()}


@torch.no_grad()
def _gradient_metrics(model, small_abs_threshold):
    """统计裁剪前梯度的 RMS、过小比例、覆盖率与非有限比例。"""
    parameters = [parameter for parameter in model.trainable_parameters()
                  if parameter.requires_grad]
    gradients = [parameter.grad.detach() for parameter in parameters
                 if parameter.grad is not None]
    total_count = sum(parameter.numel() for parameter in parameters)
    gradient_count = sum(gradient.numel() for gradient in gradients)
    if not gradients:
        zero = torch.zeros((), device=parameters[0].device if parameters else "cpu")
        return {"grad_rms": zero, "grad_small_frac": zero,
                "grad_coverage": zero, "grad_nonfinite_frac": zero}

    device = gradients[0].device
    square_sum = torch.zeros((), device=device)
    small_count = torch.zeros((), device=device)
    finite_count = torch.zeros((), device=device)
    nonfinite_count = torch.zeros((), device=device)
    for gradient in gradients:
        values = gradient.float()
        finite = torch.isfinite(values)
        finite_values = torch.where(finite, values, 0.0)
        square_sum += finite_values.square().sum()
        small_count += ((finite_values.abs() <= small_abs_threshold) & finite).sum()
        finite_count += finite.sum()
        nonfinite_count += (~finite).sum()
    finite_denominator = finite_count.clamp_min(1.0)
    gradient_denominator = max(gradient_count, 1)
    return {
        "grad_rms": (square_sum / finite_denominator).sqrt(),
        "grad_small_frac": small_count / finite_denominator,
        "grad_coverage": torch.tensor(gradient_count / max(total_count, 1), device=device),
        "grad_nonfinite_frac": nonfinite_count / gradient_denominator,
    }


def _format_losses(components):
    return "  ".join(
        ("{}={:.3e}" if name == "grad_rms" else "{}={:.4f}").format(
            name, value.detach().item())
        for name, value in components.items())
