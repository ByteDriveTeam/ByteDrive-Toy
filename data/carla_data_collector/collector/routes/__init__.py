"""由可达点构建路线队列：两两组合、按直线距离过滤、随机排序、确保不重复。公开 API 重导出入口。

模块: collector/routes/__init__.py
依赖: collector.routes.routes
读取配置: —（参数由调用方传入）
对外接口:
    - build_route_queue(spawn_points, min_d, max_d, seed, max_scenes)   # 构建路线队列
说明: 跨模块统一 `from collector.routes import ...`；实现见 routes.py，入参校验见 checks/。
"""

from collector.routes.routes import build_route_queue

__all__ = ["build_route_queue"]
