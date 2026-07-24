"""闭环模型推理与轨迹选择的公开 API 重导出入口。

模块: clone_loop/inference/__init__.py
依赖: clone_loop.inference.inference
读取配置: —（由 ClosedLoopPolicy 读取传入全局配置）
对外接口:
    - ClosedLoopPolicy(cfg)
"""

from clone_loop.inference.inference import ClosedLoopPolicy

__all__ = ["ClosedLoopPolicy"]
