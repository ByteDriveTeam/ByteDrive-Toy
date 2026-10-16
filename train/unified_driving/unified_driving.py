"""新 DrivingModel 的 Signed-Log 流匹配损失与训练循环。

模块: train/unified_driving/unified_driving.py
依赖: torch, config.schema.Config, model.driving_model.UnifiedDrivingModel
读取配置: train.precision/grad_accum_steps/grad_clip_norm/log_every
对外接口:
    - compute_flow_matching_losses(model, batch, cfg) -> (Tensor, dict)
    - train_unified_driving_epoch(model, loader, optimizer, cfg, device) -> dict
说明: 轨迹噪声插值、速度场监督和损失始终处在 Signed-Log 空间；本文件
      不调用 Signed-Log 逆变换。
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


__all__ = ["compute_flow_matching_losses", "train_unified_driving_epoch"]


_TRAJECTORY_LAYER_WEIGHTS = (0.05, 0.10, 0.15, 0.15, 0.50, 1.00)


def compute_flow_matching_losses(model, batch: Dict[str, torch.Tensor], cfg):
    """计算 BEV 与轨迹联合流匹配损失。"""
    device = batch["rgb"].device
    batch_size = int(batch["rgb"].shape[0])
    time_step = torch.rand(batch_size, device=device)
    trajectory_target = batch["trajectory"].float()
    trajectory_noise = torch.randn_like(trajectory_target)
    trajectory_state = (1.0 - time_step[:, None, None]) * trajectory_noise \
        + time_step[:, None, None] * trajectory_target
    trajectory_velocity = trajectory_target - trajectory_noise
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
        batch["rgb"], batch["intrinsics"], batch["target_point"], batch["ego_state"],
        batch["motion_matrices"], time_step=time_step, traj_state=trajectory_state,
        bev_state=bev_state)
    components = {}
    trajectory_losses = [
        ((prediction.float() - trajectory_velocity).square()).mean()
        for prediction in outputs["traj_flows"]
    ]
    components["trajectory"] = sum(weight * loss for weight, loss in zip(
        _TRAJECTORY_LAYER_WEIGHTS, trajectory_losses))
    if "bev_latent" in batch:
        components["bev"] = (outputs["bev_flow"].float() - bev_velocity).square().mean()
    else:
        components["bev"] = trajectory_target.new_zeros(())
    total = components["trajectory"] + components["bev"]
    components["total"] = total
    return total, components


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
