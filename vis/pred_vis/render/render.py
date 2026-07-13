"""渲染：把感知模型三头预测（及可选 GT）着色并合成多帧多模态对照画布。

模块: vis/pred_vis/render/render.py
依赖: cv2, numpy, vis.data_vis.palette.tag_to_bgr, vis.pred_vis.render.checks.render_checks
读取配置: —（着色量程/colormap 等由调用方以参数传入，来源为 config.pred_vis）
对外接口:
    - to_display_bgr(rgb01) -> np.ndarray                    # [3,H,W] 归一化前 RGB(0..1) -> BGR uint8
    - colorize_semantic(tag_map) -> np.ndarray               # [H,W] 语义标签 -> BGR（复用 CityScapes 调色板）
    - colorize_depth(depth_m, colormap, max_m, min_m, log_scale) -> np.ndarray # [H,W] 深度(米) -> 伪彩 BGR
    - colorize_flow(velocity, max_disp) -> np.ndarray        # [2,H,W] 速度 -> 光流经典配色 BGR
    - render_grid(rows) -> np.ndarray                        # rows=[(label,[panels])] 合成对照画布
说明: 语义复用 vis.data_vis.palette 的官方调色板（DRY）；深度/光流着色与 data_vis 同法但作用于「解码回
      物理量」的预测/GT（深度=米、光流=速度 m/s），故物理口径一致、预测与 GT 可直接对照。每行一种模态、
      每列一帧，行首标注来源（pred/gt + 模态）。合成为纯 numpy/cv2，无 torch 依赖。
"""

import cv2
import numpy as np

from vis.data_vis.palette import flow_to_bgr, tag_to_bgr
from vis.pred_vis.render.checks.render_checks import check_grid_rows

# colormap 名 -> OpenCV 常量（须与 config/schema._DATA_VIS_COLORMAPS 一致）
_COLORMAPS = {"turbo": cv2.COLORMAP_TURBO, "jet": cv2.COLORMAP_JET, "magma": cv2.COLORMAP_MAGMA,
              "viridis": cv2.COLORMAP_VIRIDIS, "plasma": cv2.COLORMAP_PLASMA,
              "inferno": cv2.COLORMAP_INFERNO}
_GUTTER = 3
_BG = (28, 28, 30)


def to_display_bgr(rgb01: np.ndarray) -> np.ndarray:
    """[3,H,W] 的 RGB(0..1) -> [H,W,3] BGR uint8（用于展示 DINO 输入去归一化后的原图）。"""
    rgb = np.clip(np.transpose(rgb01, (1, 2, 0)), 0.0, 1.0)
    bgr = (rgb[:, :, ::-1] * 255.0).astype(np.uint8)
    return np.ascontiguousarray(bgr)


def colorize_semantic(tag_map: np.ndarray) -> np.ndarray:
    """[H,W] 语义标签 -> BGR（复用官方 CityScapes 调色板的向量化映射）。"""
    height, width = tag_map.shape
    return tag_to_bgr(tag_map.reshape(-1)).reshape(height, width, 3)


def colorize_depth(depth_m: np.ndarray, colormap: str, max_m: float,
                   min_m: float, log_scale: bool) -> np.ndarray:
    """[H,W] 深度(米) -> 伪彩 BGR，按量程归一后套 colormap。

    log_scale=True 时对 [min_m, max_m] 取对数映射到 [0,1]：深度感知呈几何递增，线性着色会把近处
    挤进极窄色带；对数量程令近处占据更多色带、细节更清晰。=False 退回线性归一截断。
    """
    if log_scale:
        clipped = np.clip(depth_m, min_m, max_m)
        norm = np.log(clipped / min_m) / np.log(max_m / min_m)
    else:
        norm = np.clip(depth_m / max_m, 0.0, 1.0)
    gray = (norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(gray, _COLORMAPS[colormap])


def colorize_flow(velocity: np.ndarray, max_disp: float) -> np.ndarray:
    """[2,H,W] 速度矢量 -> BGR：委托 palette.flow_to_bgr（与 data_vis 共用唯一配色实现）。

    max_disp>0 时以该幅值(m/s)为满亮度基准；=0 则按本帧幅值 99 分位自适应。
    """
    return flow_to_bgr(velocity[0], velocity[1], max_disp)


def render_grid(rows) -> np.ndarray:
    """把 [(label, [每帧 BGR 面板])] 合成为一张对照画布：每行一模态、每列一帧。"""
    check_grid_rows(rows)
    return _vstack([_titled_row(label, panels) for label, panels in rows])


def _titled_row(label: str, panels) -> np.ndarray:
    """一行：行首面板叠加标题后与其余面板横排（面板等高等宽）。"""
    titled = [_titled(panels[0], label)] + list(panels[1:])
    sep = np.full((titled[0].shape[0], _GUTTER, 3), _BG, np.uint8)
    return cv2.hconcat([p for panel in titled for p in (sep, panel)][1:])


def _vstack(rows) -> np.ndarray:
    """各模态行间插水平间隔条后纵向拼接（每行等宽）。"""
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
