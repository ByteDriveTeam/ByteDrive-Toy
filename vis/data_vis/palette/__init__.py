"""可视化着色（语义调色板 + 光流配色）。公开 API 重导出入口。

模块: vis/data_vis/palette/__init__.py
依赖: vis.data_vis.palette.palette
读取配置: —
对外接口:
    - tag_to_bgr(tags) -> ndarray            # 语义标签→BGR 颜色的向量化映射
    - flow_to_bgr(vx, vy, max_mag) -> ndarray  # 光流/速度分量→经典光流配色 BGR
说明: 跨模块统一 `from vis.data_vis.palette import tag_to_bgr, flow_to_bgr`；实现见 palette.py（无校验）。
"""

from vis.data_vis.palette.palette import flow_to_bgr, tag_to_bgr

__all__ = ["tag_to_bgr", "flow_to_bgr"]
