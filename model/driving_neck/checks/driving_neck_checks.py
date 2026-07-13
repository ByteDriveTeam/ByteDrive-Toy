# 本文件为 model/driving_neck/driving_neck.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_neck_inputs(trunk_feat, dino_raw, trunk_channels, dino_channels):
    """校验对象: DrivingNeck.forward 入参 —— trunk/DINO 特征均 [B,C,gh,gw] 且通道、空间尺寸一致。"""
    if trunk_feat.ndim != 4 or dino_raw.ndim != 4:
        raise ValueError("trunk_feat / dino_raw 期望 (B,C,gh,gw) 四维，实际 {} / {}。".format(
            tuple(trunk_feat.shape), tuple(dino_raw.shape)))
    if int(trunk_feat.shape[1]) != trunk_channels:
        raise ValueError("trunk_feat 通道应为 {}，实际 {}。".format(trunk_channels, int(trunk_feat.shape[1])))
    if int(dino_raw.shape[1]) != dino_channels:
        raise ValueError("dino_raw 通道应为 {}，实际 {}。".format(dino_channels, int(dino_raw.shape[1])))
    if trunk_feat.shape[2:] != dino_raw.shape[2:]:
        raise ValueError("trunk_feat 与 dino_raw 空间尺寸须一致，实际 {} / {}。".format(
            tuple(trunk_feat.shape[2:]), tuple(dino_raw.shape[2:])))
