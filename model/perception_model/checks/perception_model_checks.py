# 本文件为 model/perception_model/perception_model.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_input_frames(frames, patch_size):
    """校验对象: 视觉编码器/PerceptionModel 的 frames —— 4 维 RGB 且高宽为 patch 整数倍。"""
    if frames.ndim != 4:
        raise ValueError("frames 期望 [B,3,H,W] 四维，实际 ndim={}。".format(frames.ndim))
    if int(frames.shape[1]) != 3:
        raise ValueError("frames 通道数必须为 3（RGB），实际为 {}。".format(int(frames.shape[1])))
    height, width = int(frames.shape[2]), int(frames.shape[3])
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            "frames 高宽必须为 patch_size={} 的整数倍，实际为 {}×{}。".format(patch_size, height, width)
        )
