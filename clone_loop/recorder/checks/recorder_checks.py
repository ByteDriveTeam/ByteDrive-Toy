import numpy as np


def check_video_frame(frame, height, width):
    """校验对象: _VideoSink.write.frame_bgr —— 必须是固定尺寸三通道 uint8。"""
    if frame.shape != (height, width, 3) or frame.dtype != np.uint8:
        raise ValueError(
            "录像帧期望 ({},{},3) uint8，实际 {} {}".format(
                height, width, frame.shape, frame.dtype))
