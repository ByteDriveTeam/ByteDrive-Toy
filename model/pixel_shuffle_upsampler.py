"""级联像素洗牌上采样：把低分辨率特征逐级 2× 放大回原分辨率。

模块: model/pixel_shuffle_upsampler.py
依赖: torch, model.pixel_shuffle_upsampler_checks
读取配置: —（通道调度由调用方以参数传入，来源为 config.model.heads）
对外接口:
    - PixelShuffleUpsampler(in_channels, up_channels, out_channels) -> nn.Module
        .encode(x) -> Tensor   # 前 N-1 级上采样（供 BF16 段运行）
        .decode(x) -> Tensor   # 末级上采样 + 最终解码卷积（供 FP32 段运行）
        forward(x) -> Tensor   # encode 后接 decode 的完整通路
说明: 每级为 Conv2d(C->C_out·4, 3×3) + PixelShuffle(2) + GELU：卷积升到 4 倍通道，洗牌把通道折进
      空间得 2× 分辨率、通道折回 C_out。通道随分辨率翻四倍而逐级减半，抑制高分辨率显存。
      encode/decode 的切分让调用方把「最后一次上采样 + 最终解码」单独置于 FP32：末级上采样与
      decode 卷积在 decode() 内，故外层只需对 encode 结果 .float() 再调 decode（规范：混精边界外置）。
      GELU 使级联非线性（否则纯线性洗牌可折叠）；最终解码卷积后不加激活，直接产出 logits/回归值。
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from model.pixel_shuffle_upsampler_checks import check_upsampler_args, check_upsampler_input


__all__ = ["PixelShuffleUpsampler"]


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

        # 每级：Conv2d 升到 C_out·4 → PixelShuffle(2) 折回 C_out（2× 分辨率）→ GELU
        stages = []
        current = in_channels
        for out_c in up_channels:
            stages.append(nn.Sequential(
                nn.Conv2d(current, out_c * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.GELU(),
            ))
            current = out_c
        self.stages = nn.ModuleList(stages)
        self.decode_conv = nn.Conv2d(up_channels[-1], out_channels, kernel_size=3, padding=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """前 N-1 级上采样（末级与解码留给 decode，便于外层单独置 FP32）。"""
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
