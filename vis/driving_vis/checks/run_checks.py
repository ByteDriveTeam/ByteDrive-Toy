# 本文件为 vis/driving_vis/run.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_scene_frames(selected, scene):
    """校验对象: 选中的场景帧索引 —— 目标场景须至少有一帧，否则场景名有误或数据缺失。"""
    if not selected:
        raise ValueError("场景 {} 下未找到任何帧（检查 driving_vis.scene 与数据集）。".format(scene))
