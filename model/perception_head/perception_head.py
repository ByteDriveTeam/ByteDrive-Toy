"""感知解码头：2D 残差块 + 通道压缩 + 级联像素洗牌上采样至原分辨率。

模块: model/perception_head/perception_head.py
依赖: torch, model.residual_block.ResidualBlock, model.pixel_shuffle_upsampler.PixelShuffleUpsampler,
      model.perception_head.checks.perception_head_checks
读取配置: —（通道调度由调用方以参数传入，来源为 config.model.heads）
对外接口:
    - PerceptionHead(in_channels, reduce_channels, up_channels, out_channels) -> nn.Module
        .encode(feat) -> Tensor   # 残差+压缩+上采样前段（供 BF16 段）
        .decode(x) -> Tensor      # 末级上采样+最终解码（供 FP32 段）
        forward(feat) -> Tensor   # 完整通路，输出 [B, out_channels, H, W]
说明: 单帧模型，双头（语义/深度）共用本结构，仅 out_channels 不同。encode/decode 的切分把「最后一次上采样
      与最终解码」留给 FP32：调用方在 BF16 下取 encode，再对其 .float() 调 decode（混精边界外置）。
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from model.pixel_shuffle_upsampler import PixelShuffleUpsampler
from model.perception_head.checks.perception_head_checks import check_head_features
from model.residual_block import ResidualBlock


__all__ = ["PerceptionHead"]


class PerceptionHead(nn.Module):
    """单个感知解码头。

    Args:
        in_channels: 主干输出通道（= trunk channels）。
        reduce_channels: 残差块后 1×1 压缩到的头内起始通道 C0。
        up_channels: 各级 2× 像素洗牌输出通道列表。
        out_channels: 头的输出通道数（语义=类别数 / 深度=2）。

    Shape:
        输入: `[B, in_channels, H, W]`
        输出: `[B, out_channels, H·2^L, W·2^L]`，L = len(up_channels)。
    """

    def __init__(self, in_channels: int, reduce_channels: int,
                 up_channels: List[int], out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.residual = ResidualBlock(in_channels)
        # 1×1 逐点压缩到较低的头内通道，为高分辨率上采样省显存
        self.reduce = nn.Conv2d(in_channels, reduce_channels, kernel_size=1)
        self.upsampler = PixelShuffleUpsampler(reduce_channels, up_channels, out_channels)

    def encode(self, feat: torch.Tensor) -> torch.Tensor:
        """残差 + 压缩 + 上采样前段，返回中间特征（供 BF16 段）。"""
        check_head_features(feat, self.in_channels)
        x = self.residual(feat)
        x = self.reduce(x)  # [B, C0, H, W]
        return self.upsampler.encode(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """末级上采样 + 最终解码（外层通常在 FP32 下调用），输出 [B, out, Hf, Wf]。"""
        return self.upsampler.decode(x)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """完整解码通路（不切分精度，供单精度调试/推理）。"""
        return self.decode(self.encode(feat))
