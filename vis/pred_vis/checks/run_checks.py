# 本文件为 vis/pred_vis/run.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_scene_windows(selected, scene):
    """校验对象: _select_windows 结果 —— 指定场景须至少切出一个窗口。"""
    if len(selected) == 0:
        raise ValueError("场景 {} 下无可用窗口（名称是否正确、帧数是否足够）。".format(scene))
