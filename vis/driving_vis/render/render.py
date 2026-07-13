"""渲染：把驾驶模型的三场（风险/可行驶/分布）与多模态轨迹着色，并与 RGB/Seg/Depth 合成对照画布。

模块: vis/driving_vis/render/render.py
依赖: cv2, numpy, math, data.driving_targets(BevParams/ego_xy_to_pixel),
      vis.pred_vis.render(to_display_bgr/colorize_semantic/colorize_depth),
      vis.driving_vis.render.checks.render_checks
读取配置: —（colormap / 量程等由调用方传入，来源 config.driving_vis）
对外接口:
    - colorize_field(field01, colormap, inview=None) -> np.ndarray        # [H,W] 场(0..1) -> 伪彩 BGR
    - bev_scene_composite(risk, drivable, distribution, inview) -> np.ndarray  # 三场 → RGB 合成 BEV
    - draw_trajectories(base_bgr, trajectories, confidence, gt, gt_valid, bev, draw_gt) -> np.ndarray
    - compose_canvas(rows, tile_h) -> np.ndarray                          # 混合尺寸面板 letterbox 合成
    - to_display_bgr / colorize_semantic / colorize_depth                 # 复用 pred_vis（RGB/Seg/Depth）
说明: BEV 约定与 data.driving_targets 一致（行沿 x 前向、自车在下沿中心；ego_xy_to_pixel 复用），故轨迹/场
      像素对齐。三场为 [0,1]（预测需先 sigmoid），视场外像素压暗以突出监督区。多模态轨迹从自车原点连折线，
      按置信度着色、最高置信模态加粗；GT 轨迹绿色。RGB/Seg/Depth 着色直接复用 vis.pred_vis.render（DRY）。
      compose_canvas 把不同尺寸面板按统一行高缩放、再右侧补背景对齐行宽后纵向拼接，故透视图与 BEV 图可同框。
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from data.driving_targets import BevParams, ego_xy_to_pixel
from vis.driving_vis.render.checks.render_checks import check_canvas_rows, check_field
from vis.pred_vis.render import colorize_depth, colorize_semantic, to_display_bgr


__all__ = [
    "colorize_field", "bev_scene_composite", "draw_trajectories", "compose_canvas",
    "to_display_bgr", "colorize_semantic", "colorize_depth",
]

_COLORMAPS = {"turbo": cv2.COLORMAP_TURBO, "jet": cv2.COLORMAP_JET, "magma": cv2.COLORMAP_MAGMA,
              "viridis": cv2.COLORMAP_VIRIDIS, "plasma": cv2.COLORMAP_PLASMA,
              "inferno": cv2.COLORMAP_INFERNO}
_GUTTER = 3
_BG = (28, 28, 30)
_DIM_OUTVIEW = 0.35            # 视场外像素亮度压暗系数
_TRAJ_BASE_DIM = 0.5          # 轨迹面板底图（三场合成）压暗，使轨迹线醒目
_GT_COLOR = (255, 255, 255)   # GT 轨迹（白，区别于底图可行驶绿）
_EGO_COLOR = (0, 200, 255)    # 自车标记（橙）


def colorize_field(field01: np.ndarray, colormap: str, inview: np.ndarray = None) -> np.ndarray:
    """[H,W] 场(0..1) → 伪彩 BGR；给 inview 时视场外压暗。"""
    check_field(field01, "field")
    gray = (np.clip(field01, 0.0, 1.0) * 255.0).astype(np.uint8)
    bgr = cv2.applyColorMap(gray, _COLORMAPS[colormap])
    if inview is not None:
        scale = np.where(inview[..., None] > 0, 1.0, _DIM_OUTVIEW)
        bgr = (bgr * scale).astype(np.uint8)
    return bgr


def bev_scene_composite(risk: np.ndarray, drivable: np.ndarray, distribution: np.ndarray,
                        inview: np.ndarray) -> np.ndarray:
    """三场合成一张 BEV：风险→红、可行驶→绿、轨迹分布→蓝；视场外压暗（作轨迹叠加底图）。"""
    stacked = np.stack([np.clip(distribution, 0, 1), np.clip(drivable, 0, 1),
                        np.clip(risk, 0, 1)], axis=-1)  # BGR = (B=分布, G=可行驶, R=风险)
    bgr = (stacked * 255.0).astype(np.uint8)
    scale = np.where(inview[..., None] > 0, 1.0, _DIM_OUTVIEW)
    return (bgr * scale).astype(np.uint8)


def draw_trajectories(base_bgr: np.ndarray, trajectories: np.ndarray, confidence: np.ndarray,
                      gt: np.ndarray, gt_valid: np.ndarray, bev: BevParams,
                      fov_deg: float, draw_gt: bool) -> np.ndarray:
    """在 BEV 底图上画自车、视场边界、多模态预测轨迹（按置信度着色）与可选 GT 轨迹。"""
    canvas = (base_bgr.astype(np.float32) * _TRAJ_BASE_DIM).astype(np.uint8)  # 压暗底图突出轨迹
    ego_px = _to_px(np.array([[0.0, 0.0]]), bev)[0]
    _draw_fov(canvas, bev, ego_px, fov_deg)

    if len(trajectories) > 0:                                       # 多模态预测（gt-only 面板传空）
        order = np.argsort(confidence)                             # 低分先画、最高分最后（叠在最上）
        weights = _softmax(confidence)
        best = int(np.argmax(confidence))
        for m in order:
            color = _mode_color(float(weights[m]))
            thick = 3 if m == best else 1
            _draw_path(canvas, ego_px, trajectories[m], np.ones(len(trajectories[m])), bev, color, thick)

    if draw_gt:
        _draw_path(canvas, ego_px, gt, gt_valid, bev, _GT_COLOR, 2)
    _draw_ego(canvas, ego_px)
    return canvas


def _draw_ego(canvas, ego_px) -> None:
    """自车标记：BEV 下沿中心画一个朝上（前向）的小三角。"""
    r, c = int(ego_px[0]), int(ego_px[1])
    tri = np.array([[c, r - 7], [c - 5, r + 4], [c + 5, r + 4]], np.int32)
    cv2.fillConvexPoly(canvas, tri, _EGO_COLOR)


def compose_canvas(rows, tile_h: int) -> np.ndarray:
    """把 [(label,[面板])] 合成画布：面板统一缩放到行高 tile_h，行右侧补背景对齐行宽后纵向拼接。"""
    check_canvas_rows(rows)
    built = [_titled_row(label, [_fit_h(p, tile_h) for p in panels]) for label, panels in rows]
    width = max(r.shape[1] for r in built)
    return _vstack([_pad_w(r, width) for r in built])


def _to_px(xy: np.ndarray, bev: BevParams) -> np.ndarray:
    """ego xy → [N,2] 整数像素 (row, col)。"""
    rows, cols = ego_xy_to_pixel(xy, bev)
    return np.stack((rows, cols), axis=1)


def _draw_path(canvas, ego_px, waypoints, valid, bev, color, thickness) -> None:
    """自车原点起连有效航点为折线。"""
    pts_xy = waypoints[np.asarray(valid) > 0]
    if len(pts_xy) == 0:
        return
    px = _to_px(pts_xy, bev)
    chain = np.concatenate([ego_px[None], px], axis=0)
    poly = np.stack((chain[:, 1], chain[:, 0]), axis=1).round().astype(np.int32)  # (col,row)
    cv2.polylines(canvas, [poly], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    for p in poly[1:]:
        cv2.circle(canvas, (int(p[0]), int(p[1])), max(thickness, 2), color, -1)


def _draw_fov(canvas, bev: BevParams, ego_px, fov_deg: float) -> None:
    """从自车画出前向视场两条边界线（按 x 量程延伸）。"""
    half = math.radians(fov_deg) * 0.5
    far_x = bev.x_max
    for sign in (-1.0, 1.0):
        edge = np.array([[far_x, sign * far_x * math.tan(half)]])
        px = _to_px(edge, bev)[0]
        cv2.line(canvas, (int(ego_px[1]), int(ego_px[0])), (int(px[1]), int(px[0])),
                 (90, 90, 90), 1, cv2.LINE_AA)


def _mode_color(weight: float):
    """按置信度权重着色：从暗黄到亮橙。"""
    t = float(np.clip(weight, 0.0, 1.0))
    return (int(40 + 40 * t), int(120 + 135 * t), int(200 + 55 * t))  # BGR 偏橙/黄


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _fit_h(img: np.ndarray, target_h: int) -> np.ndarray:
    """按目标行高等比缩放面板。"""
    h, w = img.shape[:2]
    new_w = max(int(round(w * target_h / h)), 1)
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_NEAREST)


def _pad_w(row: np.ndarray, width: int) -> np.ndarray:
    """右侧补背景到目标宽度。"""
    if row.shape[1] >= width:
        return row
    pad = np.full((row.shape[0], width - row.shape[1], 3), _BG, np.uint8)
    return cv2.hconcat([row, pad])


def _titled_row(label: str, panels) -> np.ndarray:
    """一行：行首面板叠标题后与其余面板横排（面板等高）。"""
    titled = [_titled(panels[0], label)] + list(panels[1:])
    sep = np.full((titled[0].shape[0], _GUTTER, 3), _BG, np.uint8)
    return cv2.hconcat([p for panel in titled for p in (sep, panel)][1:])


def _vstack(rows) -> np.ndarray:
    """行间插水平间隔条后纵向拼接（每行等宽）。"""
    if len(rows) == 1:
        return rows[0]
    sep = np.full((_GUTTER, rows[0].shape[1], 3), _BG, np.uint8)
    return cv2.vconcat([r for row in rows for r in (sep, row)][1:])


def _titled(img: np.ndarray, text: str) -> np.ndarray:
    """面板左上角标注名称（带描边，深浅背景都可读）。"""
    out = img.copy()
    cv2.putText(out, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out
