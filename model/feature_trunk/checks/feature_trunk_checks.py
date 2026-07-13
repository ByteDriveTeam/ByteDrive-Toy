# 本文件为 model/feature_trunk/feature_trunk.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_trunk_features(features, channels):
    """校验对象: FeatureTrunk.forward 入参 features —— 4 维 [B,C,H,W] 且通道等于工作维 channels。"""
    if features.ndim != 4:
        raise ValueError("features 期望 [B,C,H,W] 四维，实际 ndim={}。".format(features.ndim))
    if int(features.shape[1]) != channels:
        raise ValueError(
            "features 通道数必须为工作维 {}，实际为 {}。".format(channels, int(features.shape[1]))
        )
