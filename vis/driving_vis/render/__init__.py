"""驾驶模型可视化渲染：三场/多模态轨迹着色与混合尺寸面板合成。公开 API 重导出入口。

模块: vis/driving_vis/render/__init__.py
依赖: vis.driving_vis.render.render
读取配置: —
对外接口:
    - colorize_field / bev_scene_composite / draw_trajectories / compose_canvas
    - to_display_bgr / colorize_semantic / colorize_depth（复用 pred_vis）
说明: 跨模块统一 `from vis.driving_vis.render import ...`；实现见 render.py，校验见 checks/。
"""

from vis.driving_vis.render.render import (
    bev_scene_composite,
    colorize_depth,
    colorize_field,
    colorize_semantic,
    compose_canvas,
    draw_trajectories,
    to_display_bgr,
)

__all__ = [
    "colorize_field", "bev_scene_composite", "draw_trajectories", "compose_canvas",
    "to_display_bgr", "colorize_semantic", "colorize_depth",
]
