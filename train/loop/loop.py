"""训练与评估循环：前向 → 多任务损失 → 反向 → 梯度裁剪 → 步进，并聚合日志。

模块: train/loop/loop.py
依赖: torch, config.schema.Config, train.losses.compute_losses, train.loop.checks.loop_checks
读取配置:
    train.grad_clip_norm
    train.log_every
对外接口:
    - train_one_epoch(model, loader, optimizer, cfg, device) -> dict[str, float]      # 感知训练一轮
    - evaluate(model, loader, cfg, device) -> dict[str, float]                         # 感知无梯度评估
    - train_driving_epoch(model, loader, optimizer, cfg, device) -> dict[str, float]   # 驾驶训练一轮
    - evaluate_driving(model, loader, cfg, device) -> dict[str, float]                 # 驾驶无梯度评估
说明: 模型内部已处理 BF16/FP32 混精边界（骨干+主干+头前段 BF16，末段上采样/解码 FP32），故本循环
      不再包 autocast、直接在 FP32 下算损失。BF16 具备 FP32 指数范围，无需 GradScaler。梯度裁剪上限
      为 0 时跳过。日志聚合按样本数加权求均值。感知与驾驶两条路径共用 _LossMeter/裁剪/日志逻辑，仅前向
      输入组织与损失函数不同（驾驶前向需当前/上一帧、内外参、目标点与帧间刚性变换）。
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from config.schema import Config
from train.losses import compute_driving_losses, compute_losses
from train.loop.checks.loop_checks import check_train_inputs


__all__ = ["train_one_epoch", "evaluate", "train_driving_epoch", "evaluate_driving"]


def train_one_epoch(model, loader, optimizer, cfg: Config, device) -> Dict[str, float]:
    """训练一个 epoch，返回各损失分量的样本加权均值。"""
    check_train_inputs(model, loader, optimizer)
    model.train()
    meter = _LossMeter()

    for step, batch in enumerate(loader):
        frames, targets = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(frames)
        total, components = compute_losses(outputs, targets, cfg)

        total.backward()
        if cfg.train.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.trainable_parameters(), cfg.train.grad_clip_norm)
        optimizer.step()

        meter.update(components, int(frames.shape[0]))
        if cfg.train.log_every > 0 and step % cfg.train.log_every == 0:
            print("[train] step {} {}".format(step, _format(components)))
    return meter.averages()


@torch.no_grad()
def evaluate(model, loader, cfg: Config, device) -> Dict[str, float]:
    """无梯度评估，返回各损失分量的样本加权均值。"""
    model.eval()
    meter = _LossMeter()
    for batch in loader:
        frames, targets = _to_device(batch, device)
        _, components = compute_losses(model(frames), targets, cfg)
        meter.update(components, int(frames.shape[0]))
    return meter.averages()


def train_driving_epoch(model, loader, optimizer, cfg: Config, device) -> Dict[str, float]:
    """驾驶训练一个 epoch，返回各损失分量的样本加权均值。"""
    check_train_inputs(model, loader, optimizer)
    model.train()
    meter = _LossMeter()

    for step, batch in enumerate(loader):
        batch = _batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = _driving_forward(model, batch)
        total, components = compute_driving_losses(outputs, batch, cfg)

        total.backward()
        if cfg.train.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.trainable_parameters(), cfg.train.grad_clip_norm)
        optimizer.step()

        meter.update(components, int(batch["rgb"].shape[0]))
        if cfg.train.log_every > 0 and step % cfg.train.log_every == 0:
            print("[driving] step {} {}".format(step, _format(components)))
    return meter.averages()


@torch.no_grad()
def evaluate_driving(model, loader, cfg: Config, device) -> Dict[str, float]:
    """驾驶无梯度评估，返回各损失分量的样本加权均值。"""
    model.eval()
    meter = _LossMeter()
    for batch in loader:
        batch = _batch_to_device(batch, device)
        _, components = compute_driving_losses(_driving_forward(model, batch), batch, cfg)
        meter.update(components, int(batch["rgb"].shape[0]))
    return meter.averages()


def _driving_forward(model, batch: Dict[str, torch.Tensor]):
    """驾驶模型多输入前向：当前/上一帧 RGB + 标定 + 目标点 + 上一帧到当前帧刚性变换。"""
    return model(batch["rgb"], batch["intrinsics"], batch["extrinsics"],
                 batch["target_point"], batch["previous_rgb"],
                 batch["previous_to_current"], batch["previous_valid"])


def _batch_to_device(batch: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
    """把一个 batch 的全部张量搬到设备。"""
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _to_device(batch: Dict[str, torch.Tensor], device):
    """把一个 batch 搬到设备，拆出模型输入 frames 与监督 targets。"""
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    return batch["rgb"], batch


class _LossMeter:
    """按样本数加权累计各损失分量，便于跨步聚合出均值。"""

    def __init__(self) -> None:
        self._sums: Dict[str, float] = {}
        self._count = 0

    def update(self, components: Dict[str, torch.Tensor], n: int) -> None:
        for name, value in components.items():
            self._sums[name] = self._sums.get(name, 0.0) + value.detach().item() * n
        self._count += n

    def averages(self) -> Dict[str, float]:
        denom = max(self._count, 1)
        return {name: total / denom for name, total in self._sums.items()}


def _format(components: Dict[str, torch.Tensor]) -> str:
    return "  ".join("{}={:.4f}".format(k, v.detach().item()) for k, v in components.items())
