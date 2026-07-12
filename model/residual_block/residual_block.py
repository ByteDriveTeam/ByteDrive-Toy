"""视觉编码器使用的残差卷积模块。

模块: model/residual_block/residual_block.py
依赖: torch, model.residual_block.checks.residual_block_checks
读取配置: —（通道数与 eps 由调用方以参数传入，自身不读 config）
对外接口:
    - RMSNorm1d(normalized_shape, eps=1e-6) -> nn.Module   # 1D 序列通道 RMSNorm
    - RMSNorm2d(normalized_shape, eps=1e-6) -> nn.Module   # 2D 特征图通道 RMSNorm
    - RMSNorm3d(normalized_shape, eps=1e-6) -> nn.Module   # 3D 特征图通道 RMSNorm
    - ResidualBlock1d(channels) -> nn.Module               # 1D 无瓶颈残差卷积块
    - ResidualBlock(channels) -> nn.Module                 # 2D 瓶颈残差卷积块
    - ResidualBlock3d(channels) -> nn.Module               # 3D 瓶颈残差卷积块
    - ConvNeXtBlock3d(channels, temporal_kernel=3, spatial_kernel=5, expansion=2) -> nn.Module  # 3D ConvNeXt 块
说明: RMSNorm 只做均方根归一化不做中心化；瓶颈残差块为 1x1→3x3→GELU→1x1 结构 + 残差；
      ConvNeXt 3D 块为 深度可分离(kT×kH×kW)→RMSNorm→1x1x1 升维→GELU→1x1x1 降维 + 残差。
      RMSNorm 的均方根统计量恒在 FP32 计算再回落输入精度（BF16 下方差噪声更小；FP32 路径行为不变）。
      构造入参校验下沉到 residual_block_checks（规范 §7.1）。
"""

import torch
import torch.nn as nn

from model.residual_block.checks.residual_block_checks import (
    check_convnext_block3d,
    check_residual_channels,
    check_residual_channels_1d,
    check_rmsnorm_args,
)


__all__ = [
    "RMSNorm1d",
    "RMSNorm2d",
    "RMSNorm3d",
    "ResidualBlock1d",
    "ResidualBlock",
    "ResidualBlock3d",
    "ConvNeXtBlock3d",
]


class RMSNorm1d(nn.Module):
    """适用于 1D 序列的 RMSNorm。

    RMSNorm 只做均方根归一化，不做均值中心化。相比 LayerNorm，
    该实现计算更轻量，适合 1D 卷积序列中的通道归一化。

    Args:
        normalized_shape: 输入特征的通道数。
        eps: 数值稳定项，避免除零。

    Shape:
        输入: [N, C, L]
        输出: [N, C, L]
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        check_rmsnorm_args(normalized_shape, eps)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对 1D 序列执行 RMSNorm（均方根统计量在 FP32 计算，再回落输入精度）。"""
        orig_dtype = x.dtype
        x = x.float()
        rms = x.pow(2).mean(1, keepdim=True).sqrt()
        x = x / (rms + self.eps)
        x = self.weight[:, None].float() * x
        return x.to(orig_dtype)


class RMSNorm2d(nn.Module):
    """适用于 2D 特征图的 RMSNorm。

    RMSNorm 只做均方根归一化，不做均值中心化。相比 LayerNorm，
    该实现计算更轻量，适合卷积特征图中的通道归一化。

    Args:
        normalized_shape: 输入特征的通道数。
        eps: 数值稳定项，避免除零。

    Shape:
        输入: [N, C, H, W]
        输出: [N, C, H, W]
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        check_rmsnorm_args(normalized_shape, eps)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对 2D 特征图执行 RMSNorm（均方根统计量在 FP32 计算，再回落输入精度）。"""
        orig_dtype = x.dtype
        x = x.float()
        rms = x.pow(2).mean(1, keepdim=True).sqrt()
        x = x / (rms + self.eps)
        x = self.weight[:, None, None].float() * x
        return x.to(orig_dtype)


class RMSNorm3d(nn.Module):
    """适用于 3D 特征图的 RMSNorm。

    RMSNorm 只做均方根归一化，不做均值中心化。相比 LayerNorm，
    该实现计算更轻量，适合 3D 卷积特征图中的通道归一化。

    Args:
        normalized_shape: 输入特征的通道数。
        eps: 数值稳定项，避免除零。

    Shape:
        输入: [N, C, D, H, W]
        输出: [N, C, D, H, W]
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        check_rmsnorm_args(normalized_shape, eps)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对 3D 特征图执行 RMSNorm（均方根统计量在 FP32 计算，再回落输入精度）。"""
        orig_dtype = x.dtype
        x = x.float()
        rms = x.pow(2).mean(1, keepdim=True).sqrt()
        x = x / (rms + self.eps)
        x = self.weight[:, None, None, None].float() * x
        return x.to(orig_dtype)


class ResidualBlock1d(nn.Module):
    """1D 无瓶颈残差卷积块。

    与 2D/3D 版本不同：1D 序列通道数可能低至 1，二分瓶颈会退化，
    故这里保持全通道宽度、不做 C->C/2 压缩；中间用大核卷积（k=7）
    扩大序列感受野，首尾 1 卷积做通道混合。

    结构:
        RMSNorm -> 1 Conv(C->C) -> 7 Conv(C->C) -> GELU -> 1 Conv(C->C) + 残差连接

    Args:
        channels: 输入和输出通道数（可为 1）。

    Shape:
        输入: [N, C, L]
        输出: [N, C, L]
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        check_residual_channels_1d(channels)

        self.norm = RMSNorm1d(channels)
        self.conv1 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv2 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=7,
            stride=1,
            padding=3,
        )
        self.act = nn.GELU()
        self.conv3 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 1D 无瓶颈残差卷积。"""
        identity = x

        out = self.norm(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.act(out)
        out = self.conv3(out)

        return out + identity


class ResidualBlock(nn.Module):
    """2D 瓶颈残差卷积块。

    结构:
        RMSNorm -> 1x1 Conv(C->C/2) -> 3x3 Conv(C/2->C/2)
        -> GELU -> 1x1 Conv(C/2->C) + 残差连接

    Args:
        channels: 输入和输出通道数。

    Shape:
        输入: [N, C, H, W]
        输出: [N, C, H, W]
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        check_residual_channels(channels)
        mid_channels = channels // 2

        self.norm = RMSNorm2d(channels)
        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv2 = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.act = nn.GELU()
        self.conv3 = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 2D 瓶颈残差卷积。"""
        identity = x

        out = self.norm(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.act(out)
        out = self.conv3(out)

        return out + identity


class ResidualBlock3d(nn.Module):
    """3D 瓶颈残差卷积块。

    结构:
        RMSNorm -> 1x1x1 Conv(C->C/2) -> 3x3x3 Conv(C/2->C/2)
        -> GELU -> 1x1x1 Conv(C/2->C) + 残差连接

    Args:
        channels: 输入和输出通道数。

    Shape:
        输入: [N, C, D, H, W]
        输出: [N, C, D, H, W]
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        check_residual_channels(channels)
        mid_channels = channels // 2

        self.norm = RMSNorm3d(channels)
        self.conv1 = nn.Conv3d(
            in_channels=channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv2 = nn.Conv3d(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.act = nn.GELU()
        self.conv3 = nn.Conv3d(
            in_channels=mid_channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 3D 瓶颈残差卷积。"""
        identity = x

        out = self.norm(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.act(out)
        out = self.conv3(out)

        return out + identity


class ConvNeXtBlock3d(nn.Module):
    """3D ConvNeXt 风格卷积块（时序主干的基本单元）。

    与瓶颈残差块不同：先用深度可分离大核卷积（逐通道、时空各向异性核 kT×kH×kW）
    在低算力下扩大时空感受野，再以 1×1×1 逐点卷积升维 GELU 降维做通道混合，
    通道全程不变、便于堆叠。

    结构:
        深度可分离 Conv3d(C->C, groups=C, kT×kH×kW)
        -> RMSNorm3d -> 1×1×1 Conv(C->C·r) -> GELU -> 1×1×1 Conv(C·r->C) + 残差连接

    Args:
        channels: 输入输出通道数（深度可分离卷积逐通道，故 in=out=C）。
        temporal_kernel: 时间维核 kT（奇数，对称 padding 保持 T）。
        spatial_kernel: 空间维核 kH=kW（奇数，对称 padding 保持 H/W）。
        expansion: 逐点卷积的通道膨胀率 r。

    Shape:
        输入: [N, C, T, H, W]
        输出: [N, C, T, H, W]
    """

    def __init__(self, channels: int, temporal_kernel: int = 3,
                 spatial_kernel: int = 5, expansion: int = 2) -> None:
        super().__init__()
        check_convnext_block3d(channels, temporal_kernel, spatial_kernel, expansion)
        mid_channels = channels * expansion

        # 深度可分离：groups=channels 使每通道独立卷积；对称 padding 保持时空尺寸
        self.dwconv = nn.Conv3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(temporal_kernel, spatial_kernel, spatial_kernel),
            stride=1,
            padding=(temporal_kernel // 2, spatial_kernel // 2, spatial_kernel // 2),
            groups=channels,
        )
        self.norm = RMSNorm3d(channels)
        self.pwconv1 = nn.Conv3d(channels, mid_channels, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(mid_channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 3D ConvNeXt 卷积块。"""
        identity = x

        out = self.dwconv(x)
        out = self.norm(out)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)

        return out + identity
