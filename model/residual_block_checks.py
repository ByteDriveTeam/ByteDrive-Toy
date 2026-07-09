# 本文件为 model/residual_block.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_rmsnorm_args(normalized_shape, eps):
    """校验对象: RMSNorm1d / RMSNorm2d / RMSNorm3d 构造入参 —— 通道数为正、eps 为正。"""
    if normalized_shape <= 0:
        raise ValueError("normalized_shape 必须为正整数，实际为 {}。".format(normalized_shape))
    if eps <= 0:
        raise ValueError("eps 必须为正数，实际为 {}。".format(eps))


def check_residual_channels(channels):
    """校验对象: ResidualBlock / ResidualBlock3d 构造入参 channels —— 不小于 2（需二分瓶颈）。"""
    if channels < 2:
        raise ValueError("channels 必须不小于 2，实际为 {}。".format(channels))


def check_residual_channels_1d(channels):
    """校验对象: ResidualBlock1d 构造入参 channels —— 不小于 1（无瓶颈，不二分通道）。"""
    if channels < 1:
        raise ValueError("channels 必须不小于 1，实际为 {}。".format(channels))


def check_convnext_block3d(channels, temporal_kernel, spatial_kernel, expansion):
    """校验对象: ConvNeXtBlock3d 构造入参 —— 深度可分离逐通道，核须奇数、膨胀率为正。"""
    if channels < 1:
        raise ValueError("channels 必须不小于 1，实际为 {}。".format(channels))
    if temporal_kernel < 1 or temporal_kernel % 2 == 0:
        raise ValueError("temporal_kernel 必须为正奇数，实际为 {}。".format(temporal_kernel))
    if spatial_kernel < 1 or spatial_kernel % 2 == 0:
        raise ValueError("spatial_kernel 必须为正奇数，实际为 {}。".format(spatial_kernel))
    if expansion < 1:
        raise ValueError("expansion 必须不小于 1，实际为 {}。".format(expansion))
