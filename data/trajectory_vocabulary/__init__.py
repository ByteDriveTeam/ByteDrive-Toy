"""轨迹与速度词表模块。

模块: data/trajectory_vocabulary/__init__.py
依赖: data.trajectory_vocabulary.trajectory_vocabulary
读取配置: trajectory_vocabulary.*
对外接口:
    - generate_vocabulary(cfg) -> dict
"""

from data.trajectory_vocabulary.trajectory_vocabulary import generate_vocabulary

__all__ = ["generate_vocabulary"]
