"""采集主循环：建队列→驱动 worker→碰撞重试→读共享内存→编码+写 LMDB。公开 API 重导出入口。

模块: collector/orchestrator/__init__.py
依赖: collector.orchestrator.orchestrator
读取配置: —（实现文件读 cfg.carla_collector 全树）
对外接口:
    - run(cfg, max_scenes_override=None) -> int   # 执行采集，返回成功落盘的场景段数
说明: 跨模块统一 `from collector.orchestrator import ...`；实现见 orchestrator.py（无校验）。
"""

from collector.orchestrator.orchestrator import run

__all__ = ["run"]
