"""闭环逐步日志与汇总写入器的公开 API 重导出入口。

模块: clone_loop/logger/__init__.py
依赖: clone_loop.logger.logger
读取配置: —
对外接口:
    - RunLogger(output_root)
"""

from clone_loop.logger.logger import RunLogger

__all__ = ["RunLogger"]
