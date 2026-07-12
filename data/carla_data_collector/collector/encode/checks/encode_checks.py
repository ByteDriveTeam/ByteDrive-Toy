# 本文件为 collector/encode/encode.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_frame(frame, height, width):
    """校验对象: encode_camera 逐帧入参 frame —— 必须是 (H,W,3) 的 uint8 BGR 图。

    分辨率/编码器等数值约束已由 schema 拦截，此处只在数据进入编码器前做形状/类型一致性把关。
    """
    assert frame.ndim == 3 and frame.shape == (height, width, 3), \
        "帧形状应为 ({},{},3)，实得 {}".format(height, width, frame.shape)
    assert str(frame.dtype) == "uint8", "帧 dtype 应为 uint8，实得 {}".format(frame.dtype)
