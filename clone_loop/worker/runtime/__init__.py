"""CARLA 闭环世界生命周期的公开 API 重导出入口。

模块: clone_loop/worker/runtime/__init__.py
依赖: clone_loop.worker.runtime.runtime
读取配置: —
对外接口:
    - CarlaRuntime(cfg, shared_frame)
"""

from clone_loop.worker.runtime.runtime import CarlaRuntime

__all__ = ["CarlaRuntime"]
