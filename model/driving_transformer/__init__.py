"""重导出驾驶感知的六层位置隔离 Transformer。

模块: model/driving_transformer/__init__.py
依赖: model.driving_transformer.driving_transformer
读取配置: —
对外接口:
    - DrivingTransformer
"""

from .driving_transformer import DrivingTransformer

__all__ = ["DrivingTransformer"]
