"""级联像素洗牌上采样：把低分辨率特征逐级 2× 放大回原分辨率。

模块: model/pixel_shuffle_upsampler/pixel_shuffle_upsampler.py
依赖: torch, model.pixel_shuffle_upsampler.checks.pixel_shuffle_upsampler_checks
读取配置: —（通道调度由调用方以参数传入，来源为 config.model.heads）
对外接口:
    - PixelShuffleUpsampler(in_channels, up_channels, out_channels) -> nn.Module
        .encode(x) -> Tensor   # 前 N-1 级上采样（供 BF16 段运行）
        .decode(x) -> Tensor   # 末级上采样 + 最终解码卷积（供 FP32 段运行）
        forward(x) -> Tensor   # encode 后接 decode 的完整通路
说明: 每级为 1×1 Conv(D->4C) -> PixelShuffle(2) -> 3×3 Conv(C->C) ->
      [1×1 Conv -> SiLU -> 1×1 Conv] + 残差。首个 1×1 卷积仅做通道投影，洗牌后的 3×3
      卷积负责空间融合，随后的逐点 SiLU 残差分支负责通道混合。每一级（包括末级）都包含这个完整块。
      encode/decode 的切分让调用方把「最后一次上采样 + 最终解码」单独置于 FP32：末级上采样与
      decode 卷积在 decode() 内，故外层只需对 encode 结果 .float() 再调 decode（规范：混精边界外置）。
      SiLU 使级联非线性（否则纯线性洗牌可折叠）；最终解码卷积后不加激活，直接产出 logits/回归值。
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from model.pixel_shuffle_upsampler.checks.pixel_shuffle_upsampler_checks import (
    check_upsampler_args,
    check_upsampler_input,
)


__all__ = ["PixelShuffleUpsampler"]


class _PixelShuffleStage(nn.Module):
    """Single 2x upsampling stage with post-shuffle spatial and channel mixing."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.expand = nn.Conv2d(in_channels, out_channels * 4, kernel_size=1)
        self.shuffle = nn.PixelShuffle(2)
        self.spatial_conv = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1)
        self.channel_conv1 = nn.Conv2d(
            out_channels, out_channels, kernel_size=1)
        self.act = nn.SiLU()
        self.channel_conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_conv(self.shuffle(self.expand(x)))
        return x + self.channel_conv2(self.act(self.channel_conv1(x)))


class PixelShuffleUpsampler(nn.Module):
    """由通道调度构造的级联 2× 像素洗牌上采样器。

    Args:
        in_channels: 输入通道数 C0。
        up_channels: 各级 2× 上采样后的输出通道列表（长度 = 上采样级数）。
        out_channels: 最终解码卷积的输出通道数（不参与洗牌，故无整除约束）。

    Shape:
        输入: `[N, in_channels, H, W]`
        输出: `[N, out_channels, H·2^L, W·2^L]`，L = len(up_channels)。
    """

    def __init__(self, in_channels: int, up_channels: List[int], out_channels: int) -> None:
        super().__init__()
        check_upsampler_args(in_channels, up_channels, out_channels)
        self.in_channels = in_channels

        # 每级：1×1 投影到 4C → PixelShuffle → 3×3 空间融合 → 逐点 SiLU 残差块。
        stages = []
        current = in_channels
        for out_c in up_channels:
            stages.append(_PixelShuffleStage(current, out_c))
            current = out_c
        self.stages = nn.ModuleList(stages)

        self.decode_conv = nn.Conv2d(
            up_channels[-1], out_channels, kernel_size=3, padding=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """前 N-1 级上采样（末级与解码留给 decode，便于外层单独置 FP32）。

        每级自身已包含洗牌后的 3×3 卷积和逐点 SiLU 残差分支。
        """
        check_upsampler_input(x, self.in_channels)
        for stage in self.stages[:-1]:
            x = stage(x)
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """末级上采样 + 最终解码卷积（外层通常在 FP32 下调用本方法）。"""
        x = self.stages[-1](x)
        return self.decode_conv(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """encode 后接 decode 的完整上采样通路。"""
        return self.decode(self.encode(x))
