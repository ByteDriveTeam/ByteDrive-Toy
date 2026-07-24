"""跨解释器复用固定大小 RGB 缓冲，避免每个闭环步经 JSON 复制图像。

模块: clone_loop/shared_frame/shared_frame.py
依赖: mmap, os, pathlib, sys, clone_loop.shared_frame.checks.shared_frame_checks
读取配置: 由调用方传入 clone_loop.ipc.frame_name、相机尺寸与 clone_loop.output.root 派生的后备路径
对外接口:
    - SharedFrame(name, size_bytes, backing_path, create=False)
        .write(data) -> None
        .read() -> memoryview
        .close() -> None
说明: Windows 使用命名匿名 mmap；非 Windows 后备文件显式位于项目输出目录，避免向项目外写入。
"""

import mmap
import os
import sys
from pathlib import Path

from clone_loop.shared_frame.checks.shared_frame_checks import check_frame_args, check_frame_data


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

    def close(self):
        """关闭当前进程持有的映射句柄。"""
        self._mapping.close()
