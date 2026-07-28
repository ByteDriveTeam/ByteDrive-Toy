# 本文件为 model/bev_upsampler/bev_upsampler.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_bev_upsampler_args(in_channels, up_channels, out_channels):
    """校验对象: BevUpsampler 构造入参 —— 通道调度须为非空正整数序列。"""
    if in_channels < 1:
        raise ValueError("in_channels 必须不小于 1，实际为 {}。".format(in_channels))
    if not up_channels or any(channel < 1 for channel in up_channels):
        raise ValueError("up_channels 须为非空正整数序列，实际为 {}。".format(up_channels))
    if out_channels < 1:
        raise ValueError("out_channels 必须不小于 1，实际为 {}。".format(out_channels))


def check_bev_upsampler_input(x, in_channels):
    """校验对象: BevUpsampler.forward 入参 x —— 须为通道匹配的四维 BEV 特征。"""
    if x.ndim != 4 or int(x.shape[1]) != in_channels:
        raise ValueError("x 期望 [B,{},H,W]，实际 {}。".format(
            in_channels, tuple(x.shape)))
