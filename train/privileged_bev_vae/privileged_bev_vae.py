"""特权 BEV VAE 的监督重建、KL 正则和 epoch 训练循环。

模块: train/privileged_bev_vae/privileged_bev_vae.py
依赖: torch, train.privileged_bev_vae.checks
读取配置: train.bevseg_loss_weights, train.privileged_bev_vae.kl_weight,
          train.grad_accum_steps/log_every/grad_clip_norm/epochs
对外接口:
    - compute_privileged_bev_vae_losses(outputs, batch, cfg) -> (Tensor, dict)
    - train_privileged_bev_vae_epoch(model, loader, optimizer, cfg, device, epoch) -> dict
    - evaluate_privileged_bev_vae(model, loader, cfg, device) -> dict
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from train.bevseg import compute_bevseg_reconstruction_losses
from train.privileged_bev_vae.checks.privileged_bev_vae_checks import check_privileged_bev_vae_batch


def compute_privileged_bev_vae_losses(outputs, batch, cfg):
    """计算 BEVSeg 监督重建损失与连续高斯 latent 的 KL 正则。"""
    check_privileged_bev_vae_batch(batch)
    reconstruction, components = compute_bevseg_reconstruction_losses(
        outputs["reconstruction_logits"], batch, cfg)
    weights = cfg.train.bevseg_loss_weights
    kl = -0.5 * (1.0 + outputs["logvar"] - outputs["mu"].square() -
                 outputs["logvar"].exp()).mean()
    total = reconstruction + cfg.train.privileged_bev_vae.kl_weight * kl
    return total, {**components, "kl": kl, "total": total}


def train_privileged_bev_vae_epoch(model, loader, optimizer, cfg, device, epoch=0):
    """训练一个特权 BEV VAE epoch，并返回样本加权平均损失。"""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    sums, count = {}, 0
    steps = len(loader)
    accumulation = cfg.train.grad_accum_steps
    for step, batch in enumerate(loader):
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        total, components = compute_privileged_bev_vae_losses(
            model(batch["bevseg"], sample=True), batch, cfg)
        window = min(accumulation, steps - (step // accumulation) * accumulation)
        (total / window).backward()
        if (step + 1) % accumulation == 0 or step + 1 == steps:
            if cfg.train.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.trainable_parameters(), cfg.train.grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        batch_size = batch["bevseg"].shape[0]
        for name, value in components.items():
            contribution = value.detach() * batch_size
            sums[name] = sums.get(name, 0) + contribution
        count += batch_size
        if step == 0 or (step + 1) % cfg.train.log_every == 0 or step + 1 == steps:
            formatted = "  ".join("{}={:.4f}".format(name, value.detach().item())
                                  for name, value in components.items())
            print("[privileged_bev_vae] epoch {}/{} step {}/{} ({:.2f}%) {}".format(
                epoch + 1, cfg.train.epochs, step + 1, steps,
                (step + 1) * 100.0 / max(steps, 1), formatted), flush=True)
    return {name: (value / max(count, 1)).item() for name, value in sums.items()}


@torch.no_grad()
def evaluate_privileged_bev_vae(model, loader, cfg, device):
    """在确定性均值 latent 上评估特权 BEV VAE。"""
    model.eval()
    sums, count = {}, 0
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        _, components = compute_privileged_bev_vae_losses(model(batch["bevseg"], sample=False), batch, cfg)
        batch_size = batch["bevseg"].shape[0]
        for name, value in components.items():
            contribution = value.detach() * batch_size
            sums[name] = sums.get(name, 0) + contribution
        count += batch_size
    return {name: (value / max(count, 1)).item() for name, value in sums.items()}
