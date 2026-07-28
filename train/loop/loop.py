"""训练与评估循环：前向 → 多任务损失 → 反向 → 梯度裁剪 → 步进，并聚合日志。

模块: train/loop/loop.py
依赖: torch, config.schema.Config, train.losses.compute_losses, train.loop.checks.loop_checks
读取配置:
    train.grad_clip_norm
    train.grad_accum_steps
    train.log_every
对外接口:
    - train_one_epoch(model, loader, optimizer, cfg, device) -> dict[str, float]      # 感知训练一轮
    - evaluate(model, loader, cfg, device) -> dict[str, float]                         # 感知无梯度评估
    - train_driving_epoch(model, loader, optimizer, cfg, device) -> dict[str, float]   # 驾驶训练一轮
    - evaluate_driving(model, loader, cfg, device) -> dict[str, float]                 # 驾驶无梯度评估
说明: 模型内部已处理 BF16/FP32 混精边界（骨干+主干+头前段 BF16，末段上采样/解码 FP32），故本循环
      不再包 autocast、直接在 FP32 下算损失。BF16 具备 FP32 指数范围，无需 GradScaler。梯度裁剪上限
      为 0 时跳过。梯度按 grad_accum_steps 归一累积，最后不足窗口按实际步数归一。日志聚合在设备侧按样本数
      加权，只在打印/epoch 结束同步。感知与驾驶两条路径共用 _LossMeter/裁剪/日志逻辑，仅前向
      输入组织与损失函数不同（驾驶前向需当前/上一帧图像与 LiDAR、标定、规划条件及帧间刚性变换）。
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
    rgb_stats = _rgb_stats(cfg, device)
    accumulation = cfg.train.grad_accum_steps
    num_steps = len(loader)
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        frames, targets = _to_device(batch, device, rgb_stats)
        outputs = model(frames)
        total, components = compute_losses(outputs, targets, cfg)

        window_size = min(accumulation, num_steps - (step // accumulation) * accumulation)
        (total / window_size).backward()
        if (step + 1) % accumulation == 0 or step + 1 == num_steps:
            if cfg.train.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.trainable_parameters(), cfg.train.grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        meter.update(components, int(frames.shape[0]))
        if cfg.train.log_every > 0 and step % cfg.train.log_every == 0:
            print("[train] step {} {}".format(step, _format(components)))
    return meter.averages()


@torch.no_grad()
def evaluate(model, loader, cfg: Config, device) -> Dict[str, float]:
    """无梯度评估，返回各损失分量的样本加权均值。"""
    model.eval()
    meter = _LossMeter()
    rgb_stats = _rgb_stats(cfg, device)
    for batch in loader:
        frames, targets = _to_device(batch, device, rgb_stats)
        _, components = compute_losses(model(frames), targets, cfg)
        meter.update(components, int(frames.shape[0]))
    return meter.averages()


def train_driving_epoch(model, loader, optimizer, cfg: Config, device) -> Dict[str, float]:
    """驾驶训练一个 epoch，返回各损失分量的样本加权均值。"""
    check_train_inputs(model, loader, optimizer)
    model.train()
    meter = _LossMeter()
    rgb_stats = _rgb_stats(cfg, device)
    accumulation = cfg.train.grad_accum_steps
    num_steps = len(loader)
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        batch = _batch_to_device(batch, device, rgb_stats)
        outputs = _driving_forward(model, batch)
        total, components = compute_driving_losses(outputs, batch, cfg)

        window_size = min(accumulation, num_steps - (step // accumulation) * accumulation)
        (total / window_size).backward()
        if (step + 1) % accumulation == 0 or step + 1 == num_steps:
            if cfg.train.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.trainable_parameters(), cfg.train.grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        meter.update(components, int(batch["rgb"].shape[0]))
        if cfg.train.log_every > 0 and step % cfg.train.log_every == 0:
            print("[driving] step {} {}".format(step, _format(components)))
    return meter.averages()


@torch.no_grad()
def evaluate_driving(model, loader, cfg: Config, device) -> Dict[str, float]:
    """驾驶无梯度评估，返回各损失分量的样本加权均值。"""
    model.eval()
    meter = _LossMeter()
    rgb_stats = _rgb_stats(cfg, device)
    for batch in loader:
        batch = _batch_to_device(batch, device, rgb_stats)
        _, components = compute_driving_losses(_driving_forward(model, batch), batch, cfg)
        meter.update(components, int(batch["rgb"].shape[0]))
    return meter.averages()


def _driving_forward(model, batch: Dict[str, torch.Tensor]):
    """驾驶模型多输入前向：双帧图像/体素、标定、规划条件与帧间刚性变换。"""
    return model(batch["rgb"], batch["intrinsics"], batch["extrinsics"],
                 batch["target_point"], batch["ego_velocity"],
                 batch["previous_rgb"], batch["previous_to_current"], batch["previous_valid"],
                 lidar_stats=batch["lidar_stats"],
                 lidar_occupied=batch["lidar_occupied"],
                 lidar_valid=batch["lidar_valid"],
                 previous_lidar_stats=batch["previous_lidar_stats"],
                 previous_lidar_occupied=batch["previous_lidar_occupied"],
                 previous_lidar_valid=batch["previous_lidar_valid"])


def _batch_to_device(batch: Dict[str, torch.Tensor], device, rgb_stats) -> Dict[str, torch.Tensor]:
    """紧凑张量搬到设备后再转换图像/类别，减少锁页内存与传输量。"""
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    batch["rgb"] = _normalize_bgr(batch["rgb"], rgb_stats)
    batch["previous_rgb"] = _normalize_bgr(batch["previous_rgb"], rgb_stats)
    for name in ("lane_class", "traffic_light_state"):
        batch[name] = batch[name].long()
    batch["stop_line"] = batch["stop_line"].float()
    return batch


def _to_device(batch: Dict[str, torch.Tensor], device, rgb_stats):
    """把一个 batch 搬到设备，拆出模型输入 frames 与监督 targets。"""
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    batch["rgb"] = _normalize_bgr(batch["rgb"], rgb_stats)
    batch["semantic"] = batch["semantic"].long()
    batch["depth_inrange"] = batch["depth_inrange"].float()
    return batch["rgb"], batch


def _rgb_stats(cfg: Config, device):
    """每个 epoch 只构造一次 DINO RGB 归一化常量。"""
    mean = torch.tensor(cfg.data.dataset.dino_mean, dtype=torch.float32, device=device).view(3, 1, 1)
    std = torch.tensor(cfg.data.dataset.dino_std, dtype=torch.float32, device=device).view(3, 1, 1)
    return mean, std


def _normalize_bgr(images: torch.Tensor, rgb_stats) -> torch.Tensor:
    """设备侧 BGR uint8 → RGB FP32 → DINO 归一化，兼容单目/多目 batch。"""
    mean, std = rgb_stats
    rgb = images.flip(-3).float().div_(255.0)
    return (rgb - mean) / std


class _LossMeter:
    """按样本数加权累计各损失分量，便于跨步聚合出均值。"""

    def __init__(self) -> None:
        self._sums: Dict[str, torch.Tensor] = {}
        self._count = 0

    def update(self, components: Dict[str, torch.Tensor], n: int) -> None:
        for name, value in components.items():
            contribution = value.detach() * n
            self._sums[name] = (
                self._sums[name] + contribution if name in self._sums else contribution)
        self._count += n

    def averages(self) -> Dict[str, float]:
        denom = max(self._count, 1)
        return {name: (total / denom).item() for name, total in self._sums.items()}


def _format(components: Dict[str, torch.Tensor]) -> str:
    return "  ".join("{}={:.4f}".format(k, v.detach().item()) for k, v in components.items())
