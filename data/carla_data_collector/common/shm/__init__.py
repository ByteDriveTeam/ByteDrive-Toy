"""匿名共享内存 arena：跨进程零拷贝传大块数据，兼作场景内存缓冲。公开 API 重导出入口。

模块: common/shm/__init__.py
依赖: common.shm.shm
读取配置: —
对外接口:
    - Arena(name, size_bytes)   # 匿名共享内存区
    - BumpAllocator(arena)      # 顺序分配器（写满抛 ArenaFull）
    - ArenaFull(Exception)      # 容量不足信号
说明: 跨模块统一 `from common.shm import ...`；实现见 shm.py，入参校验见 checks/。
"""

from common.shm.shm import Arena, ArenaFull, BumpAllocator

__all__ = ["Arena", "BumpAllocator", "ArenaFull"]
