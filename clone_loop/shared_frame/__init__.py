"""固定容量传感器共享区的公开 API 重导出入口。

模块: clone_loop/shared_frame/__init__.py
依赖: clone_loop.shared_frame.shared_frame
读取配置: —
对外接口:
    - SharedFrame(name, size_bytes, backing_path, create=False)
        .write(data) / .read() / .write_prefix(data) / .read_prefix(size)
        .size_bytes / .close()
"""

from clone_loop.shared_frame.shared_frame import SharedFrame

__all__ = ["SharedFrame"]
