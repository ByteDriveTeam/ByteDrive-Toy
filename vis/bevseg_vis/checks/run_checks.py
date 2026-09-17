"""BEVSeg 数据与压缩器可视化 CLI 输入检查。"""


def check_frame(frame, history_frames, num_frames, scene_dir):
    """检查对象: main 选定的当前帧 —— 须能组成完整五帧窗口且未越界。"""
    if num_frames < history_frames:
        raise ValueError("场景不足五帧: {}".format(scene_dir))
    if frame < history_frames - 1 or frame >= num_frames:
        raise ValueError("frame 必须位于 [{}, {})".format(history_frames - 1, num_frames))
