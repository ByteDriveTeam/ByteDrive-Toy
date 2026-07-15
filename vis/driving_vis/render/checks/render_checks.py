# 本文件为 vis/driving_vis/render/render.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_field(field, name):
    """校验对象: colorize_field 入参 —— 二维 [H,W] 归一化场。"""
    if field.ndim != 2:
        raise ValueError("{} 期望二维 [H,W]，实际 {}。".format(name, tuple(field.shape)))


def check_lane_map(lane_class, lane_direction, class_colors, inview):
    """校验对象: colorize_lane_map 入参 —— 道路线类别、方向、调色板与可选视场。"""
    if lane_class.ndim != 2:
        raise ValueError("lane_class 期望二维 [H,W]，实际 {}。".format(tuple(lane_class.shape)))
    expected_direction = (2,) + tuple(lane_class.shape)
    if tuple(lane_direction.shape) != expected_direction:
        raise ValueError("lane_direction 期望 {}，实际 {}。".format(
            expected_direction, tuple(lane_direction.shape)))
    if not class_colors or any(len(color) != 3 for color in class_colors):
        raise ValueError("class_colors 需为非空 BGR 三元组序列。")
    if lane_class.size and (lane_class.min() < 0 or lane_class.max() >= len(class_colors)):
        raise ValueError("lane_class 类别索引超出 class_colors 范围。")
    if inview is not None and tuple(inview.shape) != tuple(lane_class.shape):
        raise ValueError("inview 期望 {}，实际 {}。".format(
            tuple(lane_class.shape), tuple(inview.shape)))


def check_canvas_rows(rows):
    """校验对象: compose_canvas 入参 rows —— 非空且每行至少一个面板。"""
    if not rows:
        raise ValueError("compose_canvas 需至少一行。")
    for label, panels in rows:
        if not panels:
            raise ValueError("行 '{}' 无面板。".format(label))
