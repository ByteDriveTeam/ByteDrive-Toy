# 本文件为 model/perception_head.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_head_features(feat, in_channels):
    """校验对象: PerceptionHead.encode 入参 feat —— 5 维 [B,C,T,H,W] 且通道匹配。"""
    if feat.ndim != 5:
        raise ValueError("头输入期望 [B,C,T,H,W] 五维，实际 ndim={}。".format(feat.ndim))
    if int(feat.shape[1]) != in_channels:
        raise ValueError(
            "头输入通道必须为 {}，实际为 {}。".format(in_channels, int(feat.shape[1]))
        )
