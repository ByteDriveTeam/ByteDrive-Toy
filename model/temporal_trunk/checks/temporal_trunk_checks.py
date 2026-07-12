# 本文件为 model/temporal_trunk/temporal_trunk.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_trunk_features(features, in_channels):
    """校验对象: TemporalTrunk.forward 入参 features —— 5 维 [B,Cin,T,H,W] 且投影前通道匹配。"""
    if features.ndim != 5:
        raise ValueError("features 期望 [B,Cin,T,H,W] 五维，实际 ndim={}。".format(features.ndim))
    if int(features.shape[1]) != in_channels:
        raise ValueError(
            "features 投影前通道数必须为 {}，实际为 {}。".format(in_channels, int(features.shape[1]))
        )
