"""定义 Py37 仿真 worker 与主环境闭环编排器之间的 JSON 行协议。

模块: clone_loop/protocol/protocol.py
依赖: json
读取配置: —
对外接口:
    - make_command(cmd, **args) -> dict
    - make_response(ok, result=None, error=None) -> dict
    - read_message(stream) -> dict | None
    - write_message(stream, obj) -> None
说明: RGB 大块数据固定写入共享帧区，协议只传状态、导航条件和控制量。
"""

import json


__all__ = [
    "CMD_INIT", "CMD_QUERY_ROUTES", "CMD_RESET", "CMD_STEP", "CMD_SHUTDOWN",
    "STATUS_RUNNING", "STATUS_SUCCESS", "STATUS_COLLISION", "STATUS_MAX_STEPS",
    "STATUS_OFF_ROUTE", "STATUS_MANUAL",
    "make_command", "make_response", "write_message", "read_message",
]


CMD_INIT = "init"
CMD_QUERY_ROUTES = "query_routes"
CMD_RESET = "reset"
CMD_STEP = "step"
CMD_SHUTDOWN = "shutdown"

STATUS_RUNNING = "running"
STATUS_SUCCESS = "success"
STATUS_COLLISION = "collision"
STATUS_MAX_STEPS = "max_steps"
STATUS_OFF_ROUTE = "off_route"
STATUS_MANUAL = "manual_stop"


def make_command(cmd, **args):
    """构造一条闭环控制命令。"""
    return {"cmd": cmd, "args": args}


def make_response(ok, result=None, error=None):
    """构造一条闭环控制响应。"""
    return {"ok": bool(ok), "result": result if result is not None else {}, "error": error}


def write_message(stream, obj):
    """把 JSON 对象写成单行 UTF-8，并立即刷新管道。"""
    stream.write(json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n")
    stream.flush()


def read_message(stream):
    """读取一条 JSON 响应；流已关闭时返回 None。"""
    line = stream.readline()
    if not line:
        return None
    return json.loads(line.decode("utf-8", errors="replace"))
