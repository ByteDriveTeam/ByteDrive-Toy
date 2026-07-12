"""派生并驱动 Py37 worker 子进程的控制管道客户端。公开 API 重导出入口。

模块: collector/worker_proc/__init__.py
依赖: collector.worker_proc.worker_proc
读取配置: —（python_exe 等由 orchestrator 解析后传入）
对外接口:
    - WorkerProcess(python_exe)   # 子进程控制管道客户端
说明: 跨模块统一 `from collector.worker_proc import ...`；实现见 worker_proc.py，入参校验见 checks/。
"""

from collector.worker_proc.worker_proc import WorkerProcess

__all__ = ["WorkerProcess"]
