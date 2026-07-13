# 本文件为 data/perception_dataset/perception_dataset.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_scene_root(root):
    """校验对象: PerceptionDataset 的 scene_root —— 目录须存在。"""
    if not root.is_dir():
        raise FileNotFoundError("场景根目录不存在: {}".format(root))


def check_has_frames(index, root):
    """校验对象: 逐帧索引 —— 至少有一帧可用，否则数据目录为空或不可读。"""
    if len(index) == 0:
        raise ValueError(
            "在 {} 下未能枚举到任何帧（场景为空或 LMDB 不可读）。".format(root)
        )
