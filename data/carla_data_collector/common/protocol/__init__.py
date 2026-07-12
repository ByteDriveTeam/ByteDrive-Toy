"""控制管道 JSON 行命令/响应协议与帧索引/语义Lidar dtype 定义。公开 API 重导出入口。

模块: common/protocol/__init__.py
依赖: common.protocol.protocol
读取配置: —
对外接口:
    - ProtocolError                                   # 协议异常
    - CMD_* / STATUS_* / BLOB_* / SEMANTIC_LIDAR_DTYPE  # 命令、状态、blob 字段与 lidar dtype 常量
    - make_command / make_response                    # 构造命令/响应
    - write_message / read_message                    # 收发一行 JSON 消息
说明: 跨模块统一 `from common import protocol as P`（或 `from common.protocol import ...`）；
      实现见 protocol.py，消费侧校验见 checks/protocol_checks.py。
"""

from common.protocol.protocol import (
    BLOB_DTYPE,
    BLOB_OFFSET,
    BLOB_SHAPE,
    BLOB_SIZE,
    CMD_CONTINUE_SCENE,
    CMD_INIT,
    CMD_QUERY_SPAWN_POINTS,
    CMD_SHUTDOWN,
    CMD_START_SCENE,
    ProtocolError,
    SEMANTIC_LIDAR_DTYPE,
    STATUS_COLLISION,
    STATUS_MAX_FRAMES,
    STATUS_OK,
    STATUS_PARTIAL,
    STATUS_UNREACHABLE,
    make_command,
    make_response,
    read_message,
    write_message,
)

__all__ = [
    "ProtocolError",
    "CMD_INIT", "CMD_QUERY_SPAWN_POINTS", "CMD_START_SCENE", "CMD_CONTINUE_SCENE", "CMD_SHUTDOWN",
    "SEMANTIC_LIDAR_DTYPE",
    "BLOB_OFFSET", "BLOB_SIZE", "BLOB_SHAPE", "BLOB_DTYPE",
    "STATUS_OK", "STATUS_MAX_FRAMES", "STATUS_PARTIAL", "STATUS_COLLISION", "STATUS_UNREACHABLE",
    "make_command", "make_response", "write_message", "read_message",
]
