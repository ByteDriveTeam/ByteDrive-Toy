"""渲染：把感知模型三头预测（及可选 GT）着色并合成多帧多模态对照画布。公开 API 重导出入口。

模块: vis/pred_vis/render/__init__.py
依赖: vis.pred_vis.render.render
读取配置: —（实现文件经传入的 pv 配置读取）
对外接口:
    - to_display_bgr(rgb01) -> ndarray                    # 归一化 RGB→显示 BGR
    - colorize_semantic / colorize_depth / colorize_flow  # 三模态着色
    - render_grid(rows) -> ndarray                        # 多行多模态网格合成
说明: 跨模块统一 `from vis.pred_vis import render`（或 `from vis.pred_vis.render import ...`）；
      实现见 render.py，入参校验见 checks/。
"""

from vis.pred_vis.render.render import (
    colorize_depth,
    colorize_flow,
    colorize_semantic,
    render_grid,
    to_display_bgr,
)

__all__ = [
    "to_display_bgr", "colorize_semantic", "colorize_depth", "colorize_flow", "render_grid",
]
