# 本文件为 vis/driving_vis/render/render.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_field(field, name):
    """校验对象: colorize_field 入参 —— 二维 [H,W] 归一化场。"""
    if field.ndim != 2:
        raise ValueError("{} 期望二维 [H,W]，实际 {}。".format(name, tuple(field.shape)))


def check_canvas_rows(rows):
    """校验对象: compose_canvas 入参 rows —— 非空且每行至少一个面板。"""
    if not rows:
        raise ValueError("compose_canvas 需至少一行。")
    for label, panels in rows:
        if not panels:
            raise ValueError("行 '{}' 无面板。".format(label))
