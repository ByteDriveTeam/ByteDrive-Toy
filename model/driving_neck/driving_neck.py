"""驾驶前端 neck：感知 trunk 末端特征 + DINOv3 原始特征 → RMSNorm 融合 + 深度 frustum 位置编码 + 残差。

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
说明: 双头共享的 trunk 末端特征承载深度/分割语义，DINOv3 原始特征承载通用纹理；二者激活尺度不同，各自
      RMSNorm2d 对齐后沿通道拼接，1×1 卷积融合并降到工作维 work_dim。随后叠加 frustum_encoding 的逐 patch
      几何位置特征（由内外参+深度采样反投影得），使表观特征带上「该 patch 可能落在 ego 系哪些 3D 位置」的
      先验；再过 num_residual_blocks 层 2D 瓶颈残差块在空间维提炼，产出送入 BEV 交叉注意力的图像特征。
      精度由外层 autocast 控制；frustum 内部自管几何 FP32 / MLP BF16 边界。
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
    """把感知中段表征融合为带几何先验的图像特征。

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
        """融合 + 几何位置编码 + 残差，产出图像特征 `[B, work_dim, gh, gw]`。"""
        check_neck_inputs(trunk_feat, dino_raw, self.trunk_channels, self.dino_channels)
        fused = self.fuse(torch.cat((self.norm_trunk(trunk_feat), self.norm_dino(dino_raw)), dim=1))
        fused = fused + self.frustum(fused, intrinsics, extrinsics)
        return self.res_blocks(fused)
