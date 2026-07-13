"""特征主干：已融合到工作维的单帧特征，经多层 2D 瓶颈残差块提炼空间表征。

模块: model/feature_trunk/feature_trunk.py
依赖: torch, config.schema.FeatureTrunkCfg, model.residual_block.ResidualBlock, model.feature_trunk.checks.feature_trunk_checks
读取配置:
    model.feature_trunk.channels
    model.feature_trunk.num_blocks
对外接口:
    - FeatureTrunk(cfg) -> nn.Module   # forward([B,channels,H,W]) -> [B,channels,H,W]
说明: 单帧模型，输入维已由上游 feature_fusion 对齐到工作维 channels（融合多层 DINO 特征时完成降维），
      故本模块不再自带投影，直接堆叠 num_blocks 个 2D 瓶颈残差块仅在空间维融合信息，供双头共享。
      块参数与其约束的唯一来源为 config，加载期已校验，本文件运行期仅校验入参张量通道。
      在 BF16 autocast 下运行由外层控制，本模块不强制精度。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config.schema import FeatureTrunkCfg
from model.residual_block import ResidualBlock
from model.feature_trunk.checks.feature_trunk_checks import check_trunk_features


__all__ = ["FeatureTrunk"]


class FeatureTrunk(nn.Module):
    """在工作维 channels 上堆叠 num_blocks 个 2D 瓶颈残差块的单帧特征主干。

    Args:
        cfg: 特征主干配置，唯一来源为 `config.model.feature_trunk`。

    Shape:
        输入: `[B, C, H, W]`，C = cfg.channels（上游 feature_fusion 已对齐的工作维）。
        输出: `[B, C, H, W]`，C = cfg.channels（主干内此维恒定）。
    """

    def __init__(self, cfg: FeatureTrunkCfg) -> None:
        super().__init__()
        self.cfg = cfg
        # 通道恒为 channels，逐块串联；块内校验由 ResidualBlock 自行完成
        self.blocks = nn.Sequential(*[
            ResidualBlock(cfg.channels) for _ in range(cfg.num_blocks)
        ])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """在空间维提炼特征；输入输出通道均为 cfg.channels。"""
        check_trunk_features(features, self.cfg.channels)
        return self.blocks(features)
