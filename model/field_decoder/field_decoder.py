"""三场解码头：BEV 特征经 2D 残差 + 通道压缩 + 级联像素洗牌上采样，解码为风险/可行驶/轨迹分布场。

模块: model/field_decoder/field_decoder.py
依赖: torch, config.schema.DrivingCfg, model.residual_block.ResidualBlock,
      model.pixel_shuffle_upsampler.PixelShuffleUpsampler, model.field_decoder.checks.field_decoder_checks
读取配置:
    model.driving.work_dim
    model.driving.fields.reduce_channels / up_channels / feature_channels
对外接口:
    - FieldDecoder(cfg_driving) -> nn.Module
        forward(bev_feat) -> dict[str, Tensor]   # risk / drivable / distribution，各 [B,1,Hf,Wf]
说明: 复用 PixelShuffleUpsampler（DRY）把 BEV 特征逐级 2× 放大到场分辨率（Hb·2^len(up_channels)）。上采样
      得到共享特征后接 GELU，再由三个 1×1 卷积头各自解码为一张场的 logit：风险场（遮挡及其后为风险）、
      可行驶区域场（高精地图）、轨迹分布场（使 GT 航点高分）。三场共享上采样主干、仅头不同，省显存又让空间
      结构一致。输出为未过 sigmoid 的 logit（BCE/分布损失在外层按 FP32 计算，混精边界外置）。
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from config.schema import DrivingCfg
from model.field_decoder.checks.field_decoder_checks import check_bev_feat
from model.pixel_shuffle_upsampler import PixelShuffleUpsampler
from model.residual_block import ResidualBlock


__all__ = ["FieldDecoder"]

# 三场名（顺序即输出 dict 键；场数为设计常量，非实验参数）
_FIELD_NAMES = ("risk", "drivable", "distribution")


class FieldDecoder(nn.Module):
    """把 BEV 特征上采样并解码为三张 BEV 场。

    Args:
        cfg_driving: 驾驶配置 `config.model.driving`。

    Shape:
        输入: `[B, work_dim, Hb, Wb]`；输出: dict，各场 `[B, 1, Hb·2^L, Wb·2^L]`，L=len(up_channels)。
    """

    def __init__(self, cfg_driving: DrivingCfg) -> None:
        super().__init__()
        self.work_dim = cfg_driving.work_dim
        fields = cfg_driving.fields
        self.residual = ResidualBlock(self.work_dim)
        self.reduce = nn.Conv2d(self.work_dim, fields.reduce_channels, kernel_size=1)
        self.upsampler = PixelShuffleUpsampler(
            fields.reduce_channels, fields.up_channels, fields.feature_channels)
        self.act = nn.GELU()
        # 三场各一 1×1 头；共享上采样主干
        self.heads = nn.ModuleDict(
            {name: nn.Conv2d(fields.feature_channels, 1, kernel_size=1) for name in _FIELD_NAMES})

    def forward(self, bev_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """解码三场 logit：{risk, drivable, distribution}，各 [B,1,Hf,Wf]。"""
        check_bev_feat(bev_feat, self.work_dim)
        shared = self.act(self.upsampler(self.reduce(self.residual(bev_feat))))
        return {name: head(shared) for name, head in self.heads.items()}
