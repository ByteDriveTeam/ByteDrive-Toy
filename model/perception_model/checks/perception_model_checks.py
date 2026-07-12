# 本文件为 model/perception_model/perception_model.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_input_frames(frames, patch_size):
    """校验对象: PerceptionModel.forward 入参 frames —— 5 维 [B,T,3,H,W] 且 H/W 为 patch 整数倍。"""
    if frames.ndim != 5:
        raise ValueError("frames 期望 [B,T,3,H,W] 五维，实际 ndim={}。".format(frames.ndim))
    if int(frames.shape[2]) != 3:
        raise ValueError("frames 通道数必须为 3（RGB），实际为 {}。".format(int(frames.shape[2])))
    height, width = int(frames.shape[3]), int(frames.shape[4])
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            "frames 高宽必须为 patch_size={} 的整数倍，实际为 {}×{}。".format(patch_size, height, width)
        )
