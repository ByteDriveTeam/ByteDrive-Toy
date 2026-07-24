"""闭环控制管道协议的公开 API 重导出入口。

模块: clone_loop/protocol/__init__.py
依赖: clone_loop.protocol.protocol
读取配置: —
对外接口:
    - make_command(cmd, **args) -> dict
    - make_response(ok, result=None, error=None) -> dict
    - read_message(stream) -> dict | None
    - write_message(stream, obj) -> None
"""

from clone_loop.protocol.protocol import (
    CMD_INIT,
    CMD_QUERY_ROUTES,
    CMD_RESET,
    CMD_SHUTDOWN,
    CMD_STEP,
    STATUS_COLLISION,
    STATUS_MANUAL,
    STATUS_MAX_STEPS,
    STATUS_OFF_ROUTE,
    STATUS_RUNNING,
    STATUS_SUCCESS,
    make_command,
    make_response,
    read_message,
    write_message,
)

__all__ = [
    "CMD_INIT", "CMD_QUERY_ROUTES", "CMD_RESET", "CMD_SHUTDOWN", "CMD_STEP",
    "STATUS_COLLISION", "STATUS_MANUAL", "STATUS_MAX_STEPS", "STATUS_OFF_ROUTE",
    "STATUS_RUNNING", "STATUS_SUCCESS", "make_command", "make_response", "read_message",
    "write_message",
]
