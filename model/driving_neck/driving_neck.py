"""驾驶前端 neck：融合感知 trunk 与 DINOv3 Patch 内容并作残差提炼。

模块: model/driving_neck/driving_neck.py
依赖: torch, config.schema.DrivingCfg, model.residual_block.(RMSNorm2d, ResidualBlock),
      model.frustum_encoding.FrustumEncoding, model.driving_neck.checks.driving_neck_checks
读取配置:
    model.driving.work_dim
    model.driving.neck_num_residual_blocks
    model.driving.frustum.depth_min_m / depth_max_m / step_near_m / step_far_m / coord_symlog_scale / mlp_hidden
对外接口:
    - DrivingNeck(cfg_driving, trunk_channels, dino_channels, patch_size) -> nn.Module
        forward(trunk_feat, dino_raw, intrinsics, extrinsics) -> Tensor  # [B, work_dim, gh, gw]
说明: 仅处理表观内容；frustum 只提供几何坐标，位置编码在注意力 Q/K 内使用。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config.schema import DrivingCfg
from model.driving_neck.checks.driving_neck_checks import check_neck_inputs
from model.frustum_encoding import FrustumEncoding
from model.residual_block import RMSNorm2d, ResidualBlock


__all__ = ["DrivingNeck"]


class DrivingNeck(nn.Module):
    """把感知中段表征融合为纯内容图像特征。

    Args:
        cfg_driving: 驾驶配置 `config.model.driving`。
        trunk_channels: 感知 trunk 输出通道（= feature_trunk.channels）。
        dino_channels: DINOv3 原始特征通道（= dinov3_backbone.hidden_dim）。
        patch_size: ViT patch 边长（frustum 像素反投影用）。

    Shape:
        trunk_feat: `[B, trunk_channels, gh, gw]`，dino_raw: `[B, dino_channels, gh, gw]`，
        intrinsics: `[B, 4]`，extrinsics: `[B, 6]`；输出: `[B, work_dim, gh, gw]`。
    """

    def __init__(self, cfg_driving: DrivingCfg, trunk_channels: int, dino_channels: int,
                 patch_size: int) -> None:
        super().__init__()
        self.trunk_channels = trunk_channels
        self.dino_channels = dino_channels
        d = cfg_driving.work_dim

        self.norm_trunk = RMSNorm2d(trunk_channels)
        self.norm_dino = RMSNorm2d(dino_channels)
        # 拼接后 1×1 逐点融合降到工作维
        self.fuse = nn.Conv2d(trunk_channels + dino_channels, d, kernel_size=1)

        fr = cfg_driving.frustum
        self.frustum = FrustumEncoding(
            out_dim=d, patch_size=patch_size,
            depth_min_m=fr.depth_min_m, depth_max_m=fr.depth_max_m,
            step_near_m=fr.step_near_m, step_far_m=fr.step_far_m,
            coord_symlog_scale=fr.coord_symlog_scale, mlp_hidden=fr.mlp_hidden)

        self.res_blocks = nn.Sequential(
            *(ResidualBlock(d) for _ in range(cfg_driving.neck_num_residual_blocks)))

    def forward(self, trunk_feat: torch.Tensor, dino_raw: torch.Tensor,
                intrinsics: torch.Tensor, extrinsics: torch.Tensor) -> torch.Tensor:
        """只融合视觉内容并残差提炼，产出图像特征 `[B, work_dim, gh, gw]`。"""
        check_neck_inputs(trunk_feat, dino_raw, self.trunk_channels, self.dino_channels)
        fused = self.fuse(torch.cat((self.norm_trunk(trunk_feat), self.norm_dino(dino_raw)), dim=1))
        return self.res_blocks(fused)
