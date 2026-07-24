"""CARLA 路线进度与局部目标模块的公开 API 重导出入口。

模块: clone_loop/worker/navigation/__init__.py
依赖: clone_loop.worker.navigation.navigation
读取配置: —
对外接口:
    - RouteNavigator(world_map, start, end, cfg_route)
"""

from clone_loop.worker.navigation.navigation import RouteNavigator

__all__ = ["RouteNavigator"]
