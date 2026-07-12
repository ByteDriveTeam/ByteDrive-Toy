"""时序主干：DINO 逐帧特征先 1×1×1 投影降维，堆成时序后经多层 3D ConvNeXt 块提炼时空表征。

模块: model/temporal_trunk/temporal_trunk.py
依赖: torch, config.schema.TemporalTrunkCfg, model.residual_block.ConvNeXtBlock3d, model.temporal_trunk.checks.temporal_trunk_checks
读取配置:
    model.temporal_trunk.in_channels
    model.temporal_trunk.channels
    model.temporal_trunk.num_blocks
    model.temporal_trunk.temporal_kernel
    model.temporal_trunk.spatial_kernel
    model.temporal_trunk.expansion
对外接口:
    - TemporalTrunk(cfg) -> nn.Module   # forward([B,in_channels,T,H,W]) -> [B,channels,T,H,W]
说明: 入口用 1×1×1 逐点卷积把骨干 hidden_dim（in_channels）投影到较窄的工作维 channels，
      降低后续 3D 卷积算力；投影后主干通道恒为 channels，仅在时空维融合信息，供三头共享。
      块参数与其约束的唯一来源为 config，加载期已校验，本文件运行期仅校验入参张量通道。
      在 BF16 autocast 下运行由外层控制，本模块不强制精度。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config.schema import TemporalTrunkCfg
from model.residual_block import ConvNeXtBlock3d
from model.temporal_trunk.checks.temporal_trunk_checks import check_trunk_features


__all__ = ["TemporalTrunk"]


class TemporalTrunk(nn.Module):
    """1×1×1 投影降维后堆叠 num_blocks 个 3D ConvNeXt 块的时序主干。

    Args:
        cfg: 时序主干配置，唯一来源为 `config.model.temporal_trunk`。

    Shape:
        输入: `[B, Cin, T, H, W]`，Cin = cfg.in_channels（骨干 hidden_dim）。
        输出: `[B, C, T, H, W]`，C = cfg.channels（投影后工作维）。
    """

    def __init__(self, cfg: TemporalTrunkCfg) -> None:
        super().__init__()
        self.cfg = cfg
        # 入口逐点投影：把骨干 hidden_dim 压到较窄的工作维，省下后续 3D 卷积算力
        self.proj = nn.Conv3d(cfg.in_channels, cfg.channels, kernel_size=1)
        # 投影后通道恒定，逐块串联；块内校验由 ConvNeXtBlock3d 自行完成
        self.blocks = nn.Sequential(*[
            ConvNeXtBlock3d(cfg.channels, cfg.temporal_kernel, cfg.spatial_kernel, cfg.expansion)
            for _ in range(cfg.num_blocks)
        ])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """投影降维后在时空维提炼特征；输出通道为 cfg.channels。"""
        check_trunk_features(features, self.cfg.in_channels)
        return self.blocks(self.proj(features))
