"""预测特征主干：完整继承 DINOv3 Token 序列，经三层带 patch-only 2D RoPE 的 Pre-Norm Transformer。

模块: model/feature_trunk/feature_trunk.py
依赖: torch, config.schema.FeatureTrunkCfg, model.attention.ImageSelfAttentionBlock, model.feature_trunk.checks.feature_trunk_checks
读取配置:
    model.feature_trunk.channels
    model.feature_trunk.num_layers
    model.feature_trunk.num_heads
    model.feature_trunk.mlp_ratio
    model.feature_trunk.rope_theta
对外接口:
    - FeatureTrunk(cfg) -> nn.Module   # forward([B,S,C], gh, gw) -> [B,S,C]
说明: 上游 feature_fusion 仅融合层维并降通道，Token 轴仍为 DINOv3 原始
      `[CLS, register..., patch...]` 序列。三层 Pre-Norm Transformer 对全序列作自注意力；
      RoPE 仅施加于 patch query/key，坐标从 (1,1) 起、行列步长均为 1，CLS/寄存器不编码。
      序列在本模块内不裁剪，仅在下游进入像素解码头时取出 patch 并还原网格。
      在 BF16 autocast 下运行由外层控制，本模块不强制精度。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config.schema import FeatureTrunkCfg
from model.attention import ImageSelfAttentionBlock
from model.feature_trunk.checks.feature_trunk_checks import check_trunk_features


__all__ = ["FeatureTrunk"]


class FeatureTrunk(nn.Module):
    """在完整 DINOv3 序列上堆叠三层图像 Pre-Norm Transformer。

    Args:
        cfg: 特征主干配置，唯一来源为 `config.model.feature_trunk`。

    Shape:
        输入/输出: `[B,S,C]`，C = cfg.channels，S 与 Token 顺序恒定。
    """

    def __init__(self, cfg: FeatureTrunkCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([
            ImageSelfAttentionBlock(
                cfg.channels, cfg.num_heads, cfg.mlp_ratio, cfg.rope_theta
            )
            for _ in range(cfg.num_layers)
        ])

    def forward(
        self, features: torch.Tensor, grid_height: int, grid_width: int
    ) -> torch.Tensor:
        """对完整序列作三层自注意力，仅为 patch 旋转从 (1,1) 开始的二维位置。"""
        check_trunk_features(features, self.cfg.channels, grid_height, grid_width)
        patch_positions = _patch_positions(grid_height, grid_width, features.device)
        for block in self.blocks:
            features = block(features, patch_positions)
        return features


def _patch_positions(height: int, width: int, device: torch.device) -> torch.Tensor:
    """按 DINOv3 patch 展平顺序生成二维坐标；左上 patch=(1,1)，步长为 1。"""
    rows = torch.arange(1, height + 1, device=device, dtype=torch.float32)
    columns = torch.arange(1, width + 1, device=device, dtype=torch.float32)
    row_grid, column_grid = torch.meshgrid(rows, columns, indexing="ij")
    return torch.stack((row_grid, column_grid), dim=-1).reshape(-1, 2)
