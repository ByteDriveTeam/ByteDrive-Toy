"""渲染：3D 框投影、深度/语义/光流着色、lidar+框 鸟瞰图、多面板合成与 HUD。公开 API 重导出入口。

模块: vis/data_vis/draw/__init__.py
依赖: vis.data_vis.draw.draw
读取配置: —（实现文件经传入的 vcfg 读取）
对外接口:
    - render_frame(frame, meta, vcfg, state) -> ndarray   # 合成单帧多面板画布
    - order_cameras(names) -> list                        # 相机面板排序
说明: 跨模块统一 `from vis.data_vis import draw`（或 `from vis.data_vis.draw import ...`）；实现见 draw.py（无校验）。
"""

from vis.data_vis.draw.draw import order_cameras, render_frame

__all__ = ["render_frame", "order_cameras"]
