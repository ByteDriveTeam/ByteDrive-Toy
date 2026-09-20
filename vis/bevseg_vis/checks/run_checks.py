"""BEVSeg 数据与压缩器可视化 CLI 输入检查。"""


def check_frame(frame, num_frames, scene_dir):
    """检查对象: main 选定的当前帧 —— 须位于场景帧范围内。"""
    if frame < 0 or frame >= num_frames:
        raise ValueError("frame 必须位于 [0, {})".format(num_frames))
