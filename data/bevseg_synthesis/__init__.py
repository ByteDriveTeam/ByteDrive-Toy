"""BEVSeg 驾驶语义栅格合成模块。

模块: data/bevseg_synthesis/__init__.py
依赖: data.bevseg_synthesis.bevseg_synthesis
读取配置: data.bevseg.extent_m / data.bevseg.layers
对外接口:
    - BevSegRasterizer(cfg) -> rasterizer
    - rasterize_frame(...) -> dict[str, ndarray]
"""

from data.bevseg_synthesis.bevseg_synthesis import BevSegRasterizer

__all__ = ["BevSegRasterizer"]
