# 本文件为 data/perception_dataset/perception_dataset.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_scene_root(root):
    """校验对象: PerceptionDataset 的 scene_root —— 目录须存在。"""
    if not root.is_dir():
        raise FileNotFoundError("场景根目录不存在: {}".format(root))


def check_window_fits(index, root):
    """校验对象: 开窗索引 —— 至少能切出一个窗口，否则数据/窗口配置有误。"""
    if len(index) == 0:
        raise ValueError(
            "在 {} 下未能切出任何窗口（场景为空或帧数不足 window_size）。".format(root)
        )
