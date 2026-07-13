# 本文件为 vis/pred_vis/run.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_scene_frames(selected, scene):
    """校验对象: _select_frames 结果 —— 指定场景须至少有一帧。"""
    if len(selected) == 0:
        raise ValueError("场景 {} 下无可用帧（名称是否正确、LMDB 是否可读）。".format(scene))
