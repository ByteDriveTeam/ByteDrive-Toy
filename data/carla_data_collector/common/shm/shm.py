"""匿名共享内存 arena：跨进程零拷贝传递大块传感器数据，同时充当场景内存缓冲。

模块: common/shm/shm.py
依赖: mmap, os, sys, tempfile
读取配置: —（容量/名称由调用方从 cfg.carla_collector.ipc 传入）
对外接口:
    - ArenaFull(Exception)                       # 容量不足；上层据此提前结束场景并落盘
    - Arena(name, size_bytes, create=False)      # 打开/创建共享内存区
        .write(offset, data) / .read(offset, size) / .size / .close()
    - BumpAllocator(arena)                        # writer 侧顺序分配器
        .put(data) -> (offset, size) / .reset() / .used
说明: Windows 用 mmap(-1, size, tagname=name) 走分页文件支撑的命名匿名共享内存；
      只要任一端持有句柄区域即存活，故由 collector(父进程)先创建并持有、worker(子进程)再打开。
      生产(worker 写)与消费(collector 读)不并发——场景内只写、场景结束后只读，故无需跨进程锁。
      非 Windows 平台用临时目录下的文件映射兜底（便于在 CI/本机跑离线往返测试）。
"""

import mmap
import os
import sys
import tempfile
from pathlib import Path

from common.shm.checks.shm_checks import check_arena_args, check_put

_ALIGN = 8  # 按 8 字节对齐，保证 numpy.frombuffer 对 float64 等也安全


def _align_up(n, a=_ALIGN):
    return (n + a - 1) // a * a


class ArenaFull(Exception):
    """共享内存容量不足。语义上等价于 Design ⑩ 的「内存阈值，强制结束当前场景先落盘」。"""


class Arena:
    """一段定长、按名字共享的内存区；纯数据区，元数据（偏移/形状）走控制管道。"""

    def __init__(self, name, size_bytes, create=False):
        check_arena_args(name, size_bytes)
        self._size = int(size_bytes)
        self._name = name
        self._tmp_path = None
        if sys.platform == "win32":
            # create 与 open 是同一调用：相同 tagname+size 映射到同一区域
            self._mm = mmap.mmap(-1, self._size, tagname=name)
        else:
            self._tmp_path = Path(tempfile.gettempdir()) / "{}.arena".format(name)
            fd = os.open(str(self._tmp_path), os.O_RDWR | os.O_CREAT)
            if os.path.getsize(self._tmp_path) < self._size:
                os.ftruncate(fd, self._size)
            self._mm = mmap.mmap(fd, self._size)
            os.close(fd)

    @property
    def size(self):
        return self._size

    def write(self, offset, data):
        """把 data 写到 offset；越界抛 ArenaFull（写前检查，不产生半截写入）。"""
        end = offset + len(data)
        if end > self._size:
            raise ArenaFull("写入越界: end={} > size={}".format(end, self._size))
        self._mm[offset:end] = data

    def read(self, offset, size):
        """返回 [offset, offset+size) 的 memoryview 切片（零拷贝，只读视图）。"""
        return memoryview(self._mm)[offset:offset + size]

    def close(self):
        self._mm.close()


class BumpAllocator:
    """writer 侧顺序分配器：把 bytes 依次塞入 arena，返回其 (offset, size)。

    写满即抛 ArenaFull，由采集循环捕获并以 STATUS_PARTIAL 收尾当前段（落盘后清空续采）。
    """

    def __init__(self, arena):
        self._arena = arena
        self._cursor = 0

    def put(self, data):
        """放入一块字节，返回 (offset, size)。"""
        check_put(data)
        offset = self._cursor
        self._arena.write(offset, data)  # 越界由 Arena.write 抛 ArenaFull
        self._cursor = _align_up(offset + len(data))
        return offset, len(data)

    def reset(self):
        """场景结束后复位游标，供下一场景复用同一 arena。"""
        self._cursor = 0

    @property
    def used(self):
        return self._cursor
