# 本文件为 model/pixel_shuffle_upsampler.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_upsampler_args(in_channels, up_channels, out_channels):
    """校验对象: PixelShuffleUpsampler 构造入参 —— 通道调度合法（正整数、非空）。"""
    if in_channels < 1:
        raise ValueError("in_channels 必须不小于 1，实际为 {}。".format(in_channels))
    if len(up_channels) < 1:
        raise ValueError("up_channels 至少需要一级上采样。")
    if any(c < 1 for c in up_channels):
        raise ValueError("up_channels 每级必须为正整数，实际为 {}。".format(up_channels))
    if out_channels < 1:
        raise ValueError("out_channels 必须不小于 1，实际为 {}。".format(out_channels))


def check_upsampler_input(x, in_channels):
    """校验对象: PixelShuffleUpsampler.encode 入参 x —— 4 维 [N,C,H,W] 且通道匹配。"""
    if x.ndim != 4:
        raise ValueError("上采样输入期望 [N,C,H,W] 四维，实际 ndim={}。".format(x.ndim))
    if int(x.shape[1]) != in_channels:
        raise ValueError(
            "上采样输入通道必须为 {}，实际为 {}。".format(in_channels, int(x.shape[1]))
        )
