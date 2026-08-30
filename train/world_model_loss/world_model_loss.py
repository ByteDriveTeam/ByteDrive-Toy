"""四层 Teacher 特征回归损失，含掩码权重和时空距离衰减的可见区升温。

模块: train/world_model_loss/world_model_loss.py
依赖: torch, config.schema.Config, 本模块 checks
读取配置:
    model.world_model.num_frames / patch_size / grid.cell_size_m / grid.front_m / rear_m / left_m / right_m
    train.world_model.unmasked_distance_scale_m / unmasked_time_decay_per_frame / unmasked_ramp_steps
对外接口:
    - WorldModelReconstructionLoss(cfg) -> nn.Module
说明: 掩码 Token 权重恒 1；可见 Token 按到最近空间掩码的米制距离与距最新帧的时间距离指数
      衰减，并随优化步从 0 线性升到 1。最终按权重和归一，避免升温改变总体梯度尺度。
"""

import torch
import torch.nn as nn

from config.schema import Config
from train.world_model_loss.checks.world_model_loss_checks import check_reconstruction_outputs


__all__ = ["WorldModelReconstructionLoss"]


class WorldModelReconstructionLoss(nn.Module):
    """计算四个 Teacher 层目标的带权均方误差。"""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        wm, train = cfg.model.world_model, cfg.train.world_model
        patch_height = int(round((wm.grid.front_m + wm.grid.rear_m)
                                 / wm.grid.cell_size_m)) // wm.patch_size
        patch_width = int(round((wm.grid.left_m + wm.grid.right_m)
                                / wm.grid.cell_size_m)) // wm.patch_size
        row, column = torch.meshgrid(torch.arange(patch_height), torch.arange(patch_width), indexing="ij")
        coordinates = torch.stack((row, column), dim=-1).reshape(-1, 2).float()
        token_size_m = wm.patch_size * wm.grid.cell_size_m
        self.register_buffer("spatial_distances", torch.cdist(coordinates, coordinates) * token_size_m,
                             persistent=False)
        ages = torch.arange(wm.num_frames - 1, -1, -1, dtype=torch.float32)
        self.register_buffer("time_weights", torch.exp(-ages * train.unmasked_time_decay_per_frame),
                             persistent=False)
        self.frames = int(wm.num_frames)
        self.patch_count = patch_height * patch_width
        self.distance_scale = float(train.unmasked_distance_scale_m)
        self.ramp_steps = int(train.unmasked_ramp_steps)

    def forward(self, outputs, optimizer_step):
        """返回带权总 MSE 与掩码/可见区诊断分量。"""
        check_reconstruction_outputs(outputs)
        token_mse = (outputs["predictions"].float() - outputs["targets"].float()).pow(2).mean(-1).mean(0)
        mask = outputs["mask"].reshape(-1, self.frames, self.patch_count)
        spatial_mask = mask[:, 0]
        distances = self.spatial_distances[None].masked_fill(~spatial_mask[:, None], torch.inf).min(-1).values
        visible_weight = torch.exp(-distances / self.distance_scale)
        visible_weight = visible_weight[:, None] * self.time_weights[None, :, None]
        ramp = min(float(optimizer_step) / float(self.ramp_steps), 1.0)
        weights = torch.where(mask, torch.ones_like(visible_weight), visible_weight * ramp).reshape_as(token_mse)
        total = (token_mse * weights).sum() / weights.sum().clamp_min(1.0)
        masked = token_mse[outputs["mask"]]
        visible = token_mse[~outputs["mask"]]
        return total, {
            "reconstruction": total,
            "masked_mse": masked.mean(),
            "visible_mse": visible.mean(),
            "visible_ramp": token_mse.new_tensor(ramp),
        }
