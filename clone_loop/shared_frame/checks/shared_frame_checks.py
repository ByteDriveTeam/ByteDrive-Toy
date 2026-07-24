from pathlib import Path


def check_frame_args(name, size_bytes, backing_path):
    """校验对象: SharedFrame 构造参数 —— 名称、容量与项目内后备路径必须可用。"""
    if not name or int(size_bytes) <= 0:
        raise ValueError("共享帧名称不得为空且容量必须为正")
    if not str(backing_path):
        raise ValueError("共享帧后备路径不得为空")
    if not Path(backing_path).parent.is_dir():
        raise ValueError("共享帧后备路径的父目录不存在: {}".format(Path(backing_path).parent))


def check_frame_data(data, size_bytes):
    """校验对象: SharedFrame.write 的 data —— 必须恰好填满单帧缓冲。"""
    if len(data) != size_bytes:
        raise ValueError("共享帧字节数应为 {}，实际 {}".format(size_bytes, len(data)))
