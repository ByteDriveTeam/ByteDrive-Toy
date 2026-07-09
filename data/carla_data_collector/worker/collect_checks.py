# 本文件为 worker/collect.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_destination(destination):
    """校验对象: prepare_drive 入参 destination —— 必须是带 x/y/z 的目标点对象。

    其余数值约束（max_frames_per_scene、capture_every_n_ticks 等）已由 schema 在加载期拦截。
    """
    assert destination is not None, "prepare_drive 需要一个有效的目标点 destination"
    assert all(hasattr(destination, axis) for axis in ("x", "y", "z")), \
        "destination 必须是 carla.Location（含 x/y/z）"
