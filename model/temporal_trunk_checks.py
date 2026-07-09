# 本文件为 model/temporal_trunk.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_trunk_features(features, channels):
    """校验对象: TemporalTrunk.forward 入参 features —— 5 维 [B,C,T,H,W] 且通道匹配。"""
    if features.ndim != 5:
        raise ValueError("features 期望 [B,C,T,H,W] 五维，实际 ndim={}。".format(features.ndim))
    if int(features.shape[1]) != channels:
        raise ValueError(
            "features 通道数必须为 {}，实际为 {}。".format(channels, int(features.shape[1]))
        )
