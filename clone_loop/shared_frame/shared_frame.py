"""跨解释器复用固定容量传感器缓冲，避免每个闭环步经 JSON 复制 RGB/LiDAR。

模块: clone_loop/shared_frame/shared_frame.py
依赖: mmap, os, pathlib, sys, clone_loop.shared_frame.checks.shared_frame_checks
读取配置: 由调用方传入 clone_loop.ipc.frame_name、相机尺寸与 clone_loop.output.root 派生的后备路径
对外接口:
    - SharedFrame(name, size_bytes, backing_path, create=False)
        .write(data) -> None
        .read() -> memoryview
        .write_prefix(data) -> None
        .read_prefix(size) -> memoryview
        .size_bytes -> int
        .close() -> None
说明: Windows 使用命名匿名 mmap；非 Windows 后备文件显式位于项目输出目录，避免向项目外写入。
"""

import mmap
import os
import sys
from pathlib import Path

from clone_loop.shared_frame.checks.shared_frame_checks import (
    check_frame_args,
    check_frame_data,
    check_prefix_size,
)


__all__ = ["SharedFrame"]


class SharedFrame:
    """一块固定长度的跨进程帧缓冲。"""

    def __init__(self, name, size_bytes, backing_path, create=False):
        check_frame_args(name, size_bytes, backing_path)
        self._size = int(size_bytes)
        self._file = None
        if sys.platform == "win32":
            self._mapping = mmap.mmap(-1, self._size, tagname=name)
            return
        path = Path(backing_path)
        flags = os.O_RDWR | (os.O_CREAT if create else 0)
        fd = os.open(str(path), flags)
        if create:
            os.ftruncate(fd, self._size)
        self._mapping = mmap.mmap(fd, self._size)
        os.close(fd)

    def write(self, data):
        """覆盖写入一帧；长度必须与缓冲完全一致，避免读到残留字节。"""
        check_frame_data(data, self._size)
        self._mapping[:] = data

    def read(self):
        """零拷贝返回整帧只读视图；调用方应在下一条 worker 命令前完成消费。"""
        return memoryview(self._mapping)

    def write_prefix(self, data):
        """写入变长数据前缀；有效长度由控制协议另行传递，未用尾部无需清零。"""
        check_prefix_size(len(data), self._size)
        self._mapping[:len(data)] = data

    def read_prefix(self, size):
        """零拷贝返回指定长度前缀；调用方须在下一条 worker 命令前完成复制。"""
        check_prefix_size(size, self._size)
        return memoryview(self._mapping)[:int(size)]

    @property
    def size_bytes(self):
        """返回共享区固定容量（字节）。"""
        return self._size

    def close(self):
        """关闭当前进程持有的映射句柄。"""
        self._mapping.close()
