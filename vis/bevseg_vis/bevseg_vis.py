"""从真实 CARLA LMDB 栅格化并渲染五帧 BEVSeg。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from vis.bevseg_vis.checks.bevseg_vis_checks import check_bevseg_history


__all__ = ["render_bevseg_history"]


# BGR 颜色；语义顺序来自 config.data.bevseg.layers。
_COLORS = (
    (55, 180, 55),    # drivable
    (255, 220, 40),   # lane_centerline
    (40, 165, 255),   # lane_divider
    (180, 80, 220),   # road_boundary
    (255, 150, 40),   # pedestrian_crossing
    (40, 40, 240),    # vehicle
    (80, 180, 255),   # pedestrian
    (30, 30, 255),    # stop_line_red
    (40, 190, 255),   # stop_line_yellow
    (80, 220, 80),    # stop_line_green
)
_BACKGROUND = (22, 24, 28)
_PANEL_SIZE = 192


def _title_panel(image, title, color=(235, 235, 235)):
    panel = cv2.resize(image, (_PANEL_SIZE, _PANEL_SIZE), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(panel, (0, 0), (_PANEL_SIZE - 1, 22), (10, 12, 15), -1)
    cv2.putText(panel, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                color, 1, cv2.LINE_AA)
    return panel


def _ego_marker(canvas):
    # 该函数在 256px 原图和 192px 面板上都会被调用，中心必须根据当前画布尺寸计算，不能固定使用面板尺寸。
    height, width = canvas.shape[:2]
    center = (width // 2, height // 2)
    cv2.drawMarker(canvas, center, (255, 255, 255), cv2.MARKER_CROSS, 14, 1)
    cx, cy = center
    scale = min(height, width) / 256.0
    tip = max(int(round(10 * scale)), 2)
    half = max(int(round(7 * scale)), 2)
    triangle = np.asarray([(cx, cy - tip), (cx - half, cy + half),
                           (cx + half, cy + half)], np.int32)
    cv2.fillConvexPoly(canvas, triangle, (0, 220, 255))


def _semantic_composite(semantic, layers):
    canvas = np.full((256, 256, 3), _BACKGROUND, np.uint8)
    # 可行驶底色低透明度叠加，避免遮住车道线和动态体。
    for index, color in enumerate(_COLORS[:len(layers)]):
        mask = semantic[index] > 0.5
        if not np.any(mask):
            continue
        overlay = np.zeros_like(canvas)
        overlay[mask] = color
        alpha = 0.20 if layers[index] == "drivable" else 0.82
        canvas = cv2.addWeighted(canvas, 1.0, overlay, alpha, 0.0)
    return canvas


def _direction_overlay(canvas, direction):
    result = canvas.copy()
    norm = np.linalg.norm(direction, axis=0)
    valid = norm > 0.5
    rows, cols = np.nonzero(valid)
    if not len(rows):
        return result
    # 以有效车道像素为候选，再按 16px 网格去重；直接采样网格中心会错过细车道线。
    cells_w = (direction.shape[2] + 15) // 16
    cell_ids = (rows // 16) * cells_w + cols // 16
    _, selected = np.unique(cell_ids, return_index=True)
    rows, cols = rows[selected], cols[selected]
    for row, col in zip(rows, cols):
        right, front = direction[:, row, col] / norm[row, col]
        end = (int(round(col + right * 11)), int(round(row - front * 11)))
        cv2.arrowedLine(result, (int(col), int(row)), end, (245, 245, 245), 1,
                        cv2.LINE_AA, tipLength=0.25)
    return result


def _binary_panel(mask, title, color):
    image = np.full((256, 256, 3), _BACKGROUND, np.uint8)
    image[mask > 0.5] = color
    _ego_marker(image)
    return _title_panel(image, title)


def _legend_panel(layers, width):
    """绘制颜色到语义类别的显式映射，避免只凭颜色猜测。"""
    height = 94
    panel = np.full((height, width, 3), (14, 16, 20), np.uint8)
    cv2.putText(panel, "Legend (semantic colors) | yellow triangle=ego/front | white arrows=lane direction", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA)
    columns = min(5, len(layers))
    cell_w = max(width // columns, 1)
    for index, name in enumerate(layers):
        row, column = divmod(index, columns)
        x = column * cell_w + 8
        y = 27 + row * 31
        color = tuple(int(value) for value in _COLORS[index])
        cv2.rectangle(panel, (x, y), (x + 14, y + 14), color, -1)
        cv2.putText(panel, name, (x + 20, y + 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (225, 225, 225), 1, cv2.LINE_AA)
    return panel


def render_bevseg_history(history, semantic_layers, frame_ids=None):
    """渲染五帧历史、当前语义层和方向箭头，返回 BGR 画布。"""
    check_bevseg_history(history, semantic_layers)
    frame_ids = list(range(5)) if frame_ids is None else list(frame_ids)
    if len(frame_ids) != 5:
        raise ValueError("frame_ids 必须包含五个帧号")

    prepared = [(sample["semantic"], sample["direction"]) for sample in history]
    temporal = []
    for (semantic, direction), frame_id in zip(prepared, frame_ids):
        composite = _semantic_composite(semantic, semantic_layers)
        composite = _direction_overlay(composite, direction)
        _ego_marker(composite)
        temporal.append(_title_panel(composite, "frame {}".format(frame_id)))
    temporal_row = np.hstack(temporal)

    current_semantic, current_direction = prepared[-1]
    panels = [
        _binary_panel(current_semantic[index], name, _COLORS[index])
        for index, name in enumerate(semantic_layers)
    ]
    layer_rows = [np.hstack(panels[start:start + 5]) for start in (0, 5)]
    direction = _direction_overlay(_semantic_composite(current_semantic, semantic_layers),
                                   current_direction)
    _ego_marker(direction)
    direction_panel = _title_panel(direction, "current + lane direction")
    bottom = np.hstack([layer_rows[0], direction_panel])
    bottom2 = np.hstack([layer_rows[1], np.full_like(direction_panel, _BACKGROUND)])
    width = max(temporal_row.shape[1], bottom.shape[1], bottom2.shape[1])
    def pad(row):
        if row.shape[1] == width:
            return row
        return cv2.copyMakeBorder(row, 0, 0, 0, width - row.shape[1],
                                  cv2.BORDER_CONSTANT, value=_BACKGROUND)
    canvas = np.vstack([pad(temporal_row), pad(bottom), pad(bottom2),
                        _legend_panel(semantic_layers, width)])
    cv2.putText(canvas, "BEVSeg | X=right, Y=front | ego=center", (8, canvas.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


def save_bevseg_history(history, semantic_layers, output_path, frame_ids=None):
    """渲染并保存 BEVSeg 画布。"""
    image = render_bevseg_history(history, semantic_layers, frame_ids)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise IOError("无法写入 BEVSeg 可视化: {}".format(output_path))
    return output_path
