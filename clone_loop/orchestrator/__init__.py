"""闭环 episode 编排器的公开 API 重导出入口。

模块: clone_loop/orchestrator/__init__.py
依赖: clone_loop.orchestrator.orchestrator
读取配置: —
对外接口:
    - run_closed_loop(cfg, max_episodes_override=None) -> dict
"""

from clone_loop.orchestrator.orchestrator import run_closed_loop

__all__ = ["run_closed_loop"]
