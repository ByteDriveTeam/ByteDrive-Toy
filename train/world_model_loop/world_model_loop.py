"""世界模型单轮联合训练：重建梯度累积与 VISReg 有效 batch 两遍梯度缓存。

模块: train/world_model_loop/world_model_loop.py
依赖: contextlib, torch, config.schema.Config, model.world_model, train.visreg,
      train.world_model_loss, train.gradient_monitor, 本模块 checks
读取配置:
    train.world_model.grad_accum_steps / grad_clip_norm / log_every / amp_dtype
    train.world_model.reconstruction_weight / visreg_weight
    train.world_model.visreg.* / grad_monitor.* / 无掩码损失参数（由子模块读取）
对外接口:
    - train_world_model_epoch(model, loader, optimizer, cfg, device, global_step, epoch_seed)
        -> tuple[dict,int]
说明: 先 no_grad 汇总累计窗口内同一次重建掩码对应的 Student 末端 GAP，在完整有效 batch 上
      求 VISReg 对 GAP 的梯度，再以相同掩码逐微批重放，联合反传重建目标与缓存梯度。该过程
      等价于一次大 batch VISReg 反传，却不跨微批保留 Encoder 计算图。
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn

from config.schema import Config
from model.world_model import sample_consistent_mask
from train.gradient_monitor import monitor_gradients
from train.visreg import VISRegLoss
from train.world_model_loop.checks.world_model_loop_checks import check_world_model_train_inputs
from train.world_model_loss import WorldModelReconstructionLoss


__all__ = ["train_world_model_epoch"]


def train_world_model_epoch(model, loader, optimizer, cfg: Config, device,
                            global_step: int, epoch_seed: int):
    """联合训练一个 epoch，返回样本加权统计与更新后的优化步数。"""
    check_world_model_train_inputs(model, loader, optimizer)
    model.train()
    train_cfg = cfg.train.world_model
    reconstruction_loss = WorldModelReconstructionLoss(cfg).to(device)
    visreg_loss = VISRegLoss(cfg).to(device)
    meter = _Meter()
    generator = torch.Generator(device=device).manual_seed(epoch_seed)
    optimizer.zero_grad(set_to_none=True)
    for window in _windows(loader, train_cfg.grad_accum_steps):
        if train_cfg.visreg_weight > 0 and _sample_count(window) >= 2:
            components = _joint_window(
                model, window, reconstruction_loss, visreg_loss, cfg, device,
                global_step, generator)
        else:
            components = _reconstruction_window(
                model, window, reconstruction_loss, cfg, device, global_step, generator)
            if train_cfg.visreg_weight > 0:
                components["visreg_skipped_small_batch"] = torch.ones((), device=device)
        gradient = monitor_gradients(model, cfg, global_step)
        if cfg.train.world_model.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(
                model.trainable_parameters(), cfg.train.world_model.grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model.update_teacher()
        batch_size = sum(int(batch["grid"].shape[0]) for batch in window)
        meter.update({**components, **gradient}, batch_size)
        if train_cfg.log_every > 0 and global_step % train_cfg.log_every == 0:
            print("[world-model] step={} {}".format(
                global_step, _format({**components, **gradient})))
        global_step += 1
    return meter.averages(), global_step


def _reconstruction_window(model, window, loss_fn, cfg, device, step, generator):
    component_sums = {}
    total_count = _sample_count(window)
    for batch in window:
        grids = batch["grid"].to(device, non_blocking=True)
        mask = sample_consistent_mask(len(grids), cfg, device, generator)
        with _autocast(cfg, device):
            outputs = model.forward_reconstruction(grids, mask)
            loss, components = loss_fn(outputs, step)
            scale = len(grids) / total_count
            scaled = cfg.train.world_model.reconstruction_weight * loss * scale
        scaled.backward()
        _sum_components(component_sums, components, scale)
    return component_sums


def _joint_window(model, window, reconstruction_fn, visreg_fn, cfg, device,
                  step, generator):
    masks, gaps = [], []
    with torch.no_grad():
        for batch in window:
            grids = batch["grid"].to(device, non_blocking=True)
            mask = sample_consistent_mask(len(grids), cfg, device, generator)
            with _autocast(cfg, device):
                gaps.append(model.encode_gap(grids, mask))
            masks.append(mask)
    proxy = torch.cat(gaps).float().detach().requires_grad_(True)
    visreg, visreg_components = visreg_fn(proxy)
    effective_visreg = cfg.train.world_model.visreg_weight * visreg
    feature_gradient = torch.autograd.grad(effective_visreg, proxy)[0]

    reconstruction_components = {}
    offset = 0
    total_count = _sample_count(window)
    for batch, mask in zip(window, masks):
        grids = batch["grid"].to(device, non_blocking=True)
        count = len(grids)
        with _autocast(cfg, device):
            outputs = model.forward_reconstruction(grids, mask)
            reconstruction, components = reconstruction_fn(outputs, step)
            scale = count / total_count
            objective = (cfg.train.world_model.reconstruction_weight * reconstruction * scale
                         + (outputs["student_gap"].float()
                            * feature_gradient[offset:offset + count]).sum())
            _sum_components(reconstruction_components, components, scale)
        objective.backward()
        offset += count
    return {
        **reconstruction_components,
        **{name: value.detach() for name, value in visreg_components.items()},
        "visreg": visreg.detach(),
    }


def _windows(loader, size):
    window = []
    for batch in loader:
        window.append(batch)
        if len(window) == size:
            yield window
            window = []
    if window:
        yield window


def _sample_count(window):
    return sum(int(batch["grid"].shape[0]) for batch in window)


def _autocast(cfg, device):
    enabled = device.type == "cuda" and cfg.train.world_model.amp_dtype == "bfloat16"
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if enabled else nullcontext()


def _sum_components(target, components, scale):
    for name, value in components.items():
        contribution = value.detach() * scale
        target[name] = target.get(name, contribution.new_zeros(())) + contribution


class _Meter:
    def __init__(self):
        self.sums = {}
        self.count = 0

    def update(self, components, count):
        for name, value in components.items():
            contribution = value.detach() * count
            self.sums[name] = self.sums.get(name, contribution.new_zeros(())) + contribution
        self.count += count

    def averages(self):
        return {name: float(value / max(self.count, 1)) for name, value in self.sums.items()}


def _format(components):
    return "  ".join("{}={:.4g}".format(name, float(value)) for name, value in components.items())
