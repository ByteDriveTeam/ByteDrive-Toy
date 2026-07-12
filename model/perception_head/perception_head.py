"""感知解码头：3D 残差块 + 通道压缩 + 级联像素洗牌上采样至原分辨率。

模块: model/perception_head/perception_head.py
依赖: torch, model.residual_block.ResidualBlock3d, model.pixel_shuffle_upsampler.PixelShuffleUpsampler,
      model.perception_head.checks.perception_head_checks
读取配置: —（通道调度由调用方以参数传入，来源为 config.model.heads）
对外接口:
    - PerceptionHead(in_channels, reduce_channels, up_channels, out_channels) -> nn.Module
        .encode(feat) -> (Tensor, (B, T))   # 残差+压缩+折叠时序+上采样前段（供 BF16 段）
        .decode(x, meta) -> Tensor          # 末级上采样+解码+还原时序（供 FP32 段）
        forward(feat) -> Tensor             # 完整通路，输出 [B, out_channels, T, H, W]
说明: 三头（语义/光流/深度）共用本结构，仅 out_channels 不同。上采样对每帧独立进行，故把时序 T
      并入 batch 维（[B,C,T,H,W]->[B*T,C,H,W]）再展开还原。encode/decode 的切分把「最后一次上采样
      与最终解码」留给 FP32：调用方在 BF16 下取 encode，再对其 .float() 调 decode（混精边界外置）。
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from model.pixel_shuffle_upsampler import PixelShuffleUpsampler
from model.perception_head.checks.perception_head_checks import check_head_features
from model.residual_block import ResidualBlock3d


__all__ = ["PerceptionHead"]


class PerceptionHead(nn.Module):
    """单个感知解码头。

    Args:
        in_channels: 主干输出通道（= trunk channels）。
        reduce_channels: 残差块后 1×1×1 压缩到的头内起始通道 C0。
        up_channels: 各级 2× 像素洗牌输出通道列表。
        out_channels: 头的输出通道数（语义=类别数 / 光流=2 / 深度=2）。

    Shape:
        输入: `[B, in_channels, T, H, W]`
        输出: `[B, out_channels, T, H·2^L, W·2^L]`，L = len(up_channels)。
    """

    def __init__(self, in_channels: int, reduce_channels: int,
                 up_channels: List[int], out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.residual = ResidualBlock3d(in_channels)
        # 1×1×1 逐点压缩到较低的头内通道，为高分辨率上采样省显存
        self.reduce = nn.Conv3d(in_channels, reduce_channels, kernel_size=1)
        self.upsampler = PixelShuffleUpsampler(reduce_channels, up_channels, out_channels)

    def encode(self, feat: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """残差 + 压缩 + 折叠时序 + 上采样前段；返回中间特征与 (B, T) 供还原。"""
        check_head_features(feat, self.in_channels)
        batch, _, frames = int(feat.shape[0]), int(feat.shape[1]), int(feat.shape[2])

        x = self.residual(feat)
        x = self.reduce(x)  # [B, C0, T, H, W]
        # 上采样逐帧独立：把 T 并入 batch。permute 后 reshape 自动拷成连续
        c0, height, width = int(x.shape[1]), int(x.shape[3]), int(x.shape[4])
        x = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, c0, height, width)
        x = self.upsampler.encode(x)
        return x, (batch, frames)

    def decode(self, x: torch.Tensor, meta: Tuple[int, int]) -> torch.Tensor:
        """末级上采样 + 最终解码，并把时序从 batch 维还原（外层通常在 FP32 下调用）。"""
        batch, frames = meta
        x = self.upsampler.decode(x)  # [B*T, out, Hf, Wf]
        out_c, height, width = int(x.shape[1]), int(x.shape[2]), int(x.shape[3])
        # [B*T, out, Hf, Wf] -> [B, out, T, Hf, Wf]
        return x.reshape(batch, frames, out_c, height, width).permute(0, 2, 1, 3, 4).contiguous()

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """完整解码通路（不切分精度，供单精度调试/推理）。"""
        x, meta = self.encode(feat)
        return self.decode(x, meta)
