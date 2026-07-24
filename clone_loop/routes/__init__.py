"""闭环路线队列构造的公开 API 重导出入口。

模块: clone_loop/routes/__init__.py
依赖: clone_loop.routes.routes
读取配置: —
对外接口:
    - build_route_queue(spawn_points, min_distance, max_distance, seed, max_episodes) -> list[dict]
"""

from clone_loop.routes.routes import build_route_queue

__all__ = ["build_route_queue"]
