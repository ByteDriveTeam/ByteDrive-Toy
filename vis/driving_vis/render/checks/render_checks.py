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


def check_traffic_control(stop_line, state_map, state_valid, state_colors, unknown_color, inview):
    """校验对象: colorize_traffic_control 入参 —— 停止线、灯态、有效掩码、颜色与视场。"""
    if stop_line.ndim != 2:
        raise ValueError("stop_line 期望二维 [H,W]，实际 {}。".format(tuple(stop_line.shape)))
    if tuple(state_map.shape) != tuple(stop_line.shape):
        raise ValueError("state_map 期望 {}，实际 {}。".format(
            tuple(stop_line.shape), tuple(state_map.shape)))
    if state_valid is not None and tuple(state_valid.shape) != tuple(stop_line.shape):
        raise ValueError("state_valid 期望 {}，实际 {}。".format(
            tuple(stop_line.shape), tuple(state_valid.shape)))
    if not state_colors or any(len(color) != 3 for color in state_colors) or len(unknown_color) != 3:
        raise ValueError("state_colors 与 unknown_color 需为 BGR 三元组。")
    if state_map.size and (state_map.min() < 0 or state_map.max() >= len(state_colors)):
        raise ValueError("state_map 类别索引超出 state_colors 范围。")
    if inview is not None and tuple(inview.shape) != tuple(stop_line.shape):
        raise ValueError("inview 期望 {}，实际 {}。".format(
            tuple(stop_line.shape), tuple(inview.shape)))


def check_traffic_overlay(base_bgr, traffic_bgr, stop_mask, alpha):
    """校验对象: overlay_traffic_control 入参 —— 等尺寸 BGR 底图、交通控制图与停止线掩码。"""
    if base_bgr.ndim != 3 or base_bgr.shape[2] != 3:
        raise ValueError("base_bgr 期望 [H,W,3]。")
    if tuple(traffic_bgr.shape) != tuple(base_bgr.shape):
        raise ValueError("traffic_bgr 期望 {}，实际 {}。".format(
            tuple(base_bgr.shape), tuple(traffic_bgr.shape)))
    if tuple(stop_mask.shape) != tuple(base_bgr.shape[:2]):
        raise ValueError("stop_mask 期望 {}，实际 {}。".format(
            tuple(base_bgr.shape[:2]), tuple(stop_mask.shape)))
    if not 0 < alpha <= 1:
        raise ValueError("alpha 必须在 (0,1]。")


def check_canvas_rows(rows):
    """校验对象: compose_canvas 入参 rows —— 非空且每行至少一个面板。"""
    if not rows:
        raise ValueError("compose_canvas 需至少一行。")
    for label, panels in rows:
        if not panels:
            raise ValueError("行 '{}' 无面板。".format(label))
