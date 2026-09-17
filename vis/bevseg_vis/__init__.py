"""BEVSeg 数据、压缩器重建与量化特征 PCA 可视化公共 API。

模块: vis/bevseg_vis/__init__.py
依赖: vis.bevseg_vis.bevseg_vis
读取配置: —
对外接口:
    - render_bevseg_history(...) -> ndarray
    - save_bevseg_history(...) -> Path
    - render_bevseg_reconstruction(...) -> ndarray
    - save_bevseg_reconstruction(...) -> Path
"""

from vis.bevseg_vis.bevseg_vis import (
    render_bevseg_history,
    render_bevseg_reconstruction,
    save_bevseg_history,
    save_bevseg_reconstruction,
)

__all__ = [
    "render_bevseg_history",
    "save_bevseg_history",
    "render_bevseg_reconstruction",
    "save_bevseg_reconstruction",
]
