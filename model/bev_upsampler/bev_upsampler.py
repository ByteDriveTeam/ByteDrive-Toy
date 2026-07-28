"""BEV 专用像素洗牌上采样器：以空间卷积和激活残差逐级恢复高分辨率特征。

模块: model/bev_upsampler/bev_upsampler.py
依赖: torch, model.bev_upsampler.checks.bev_upsampler_checks
读取配置: —
对外接口:
    - BevUpsampler(in_channels, up_channels, out_channels) -> nn.Module
        forward(x) -> Tensor   # 逐级 2× 上采样后的 BEV 特征
说明: 每级采用 3×3 Conv → PixelShuffle → SiLU → 1×1 Conv，并从 PixelShuffle 输出、
      SiLU 之前引出残差。该模块仅供驾驶 BEV 解码分支使用，参数路径与感知上采样器隔离，
      使旧视场检查点兼容恢复时自动重新初始化 BEV 上采样权重。
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from model.bev_upsampler.checks.bev_upsampler_checks import (
    check_bev_upsampler_args,
    check_bev_upsampler_input,
)


__all__ = ["BevUpsampler"]


class _BevUpsampleStage(nn.Module):
    """单级 2× BEV 上采样与激活残差。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.spatial_expand = nn.Conv2d(
            in_channels, out_channels * 4, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        self.act = nn.SiLU()
        self.channel_projection = nn.Conv2d(
            out_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shuffle(self.spatial_expand(x))
        return residual + self.channel_projection(self.act(residual))


class BevUpsampler(nn.Module):
    """逐级放大 BEV 特征，并投影到预测头所需通道数。

    参数:
        in_channels: 输入 BEV 特征通道数
        up_channels: 每级 2× 上采样后的通道数
        out_channels: 末端共享预测特征通道数
    返回:
        形状为 `[B, out_channels, H·2^L, W·2^L]` 的张量，L 为上采样级数
    """

    def __init__(self, in_channels: int, up_channels: List[int],
                 out_channels: int) -> None:
        super().__init__()
        check_bev_upsampler_args(in_channels, up_channels, out_channels)
        self.in_channels = in_channels
        channels = [in_channels, *up_channels]
        self.stages = nn.Sequential(*(
            _BevUpsampleStage(source, target)
            for source, target in zip(channels, up_channels)
        ))
        self.output_projection = nn.Conv2d(
            up_channels[-1], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回逐级上采样并完成通道投影的 BEV 特征。"""
        check_bev_upsampler_input(x, self.in_channels)
        return self.output_projection(self.stages(x))
