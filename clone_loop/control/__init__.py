"""轨迹跟踪控制器的公开 API 重导出入口。

模块: clone_loop/control/__init__.py
依赖: clone_loop.control.control
读取配置: —（由 TrajectoryController 读取传入的 clone_loop.control）
对外接口:
    - TrajectoryController(cfg_control, fixed_delta_seconds)
"""

from clone_loop.control.control import TrajectoryController

__all__ = ["TrajectoryController"]
