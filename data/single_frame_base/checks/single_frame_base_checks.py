# 本文件为 data/single_frame_base/single_frame_base.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_scene_root(root):
    """校验对象: SingleFrameSceneBase 构造的 scene_root —— 目录须存在。"""
    if not root.is_dir():
        raise FileNotFoundError("场景根目录不存在: {}（先运行采集或检查 config 中的 scene_root）。".format(root))


def check_has_frames(index, root):
    """校验对象: SingleFrameSceneBase 帧索引 —— 至少有一帧可用，否则数据集为空。"""
    if not index:
        raise ValueError("在 {} 下未找到任何可用帧（场景 LMDB 缺失或 num_frames 为 0）。".format(root))
