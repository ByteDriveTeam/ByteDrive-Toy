"""Py37 CARLA 闭环 worker CLI：接收 JSON 命令、推进仿真并把 RGB 写入共享帧区。

模块: clone_loop/worker/run.py
依赖: os, sys, traceback, pathlib, config.schema, clone_loop.protocol,
      clone_loop.shared_frame, clone_loop.worker.runtime
读取配置: 不读配置文件；主进程以 init 命令下发完整配置字典
对外接口:
    - main() -> None
说明: stdout 独占协议；Python/C 扩展日志整体重定向到 stderr，避免污染 JSON 行。
"""

import os
import sys
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]
_COLLECTOR_ROOT = _REPO_ROOT / "data" / "carla_data_collector"
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_COLLECTOR_ROOT))

from config.schema import build_config
from clone_loop import protocol as P
from clone_loop.protocol import make_response, read_message, write_message
from clone_loop.shared_frame import SharedFrame
from clone_loop.worker.runtime import CarlaRuntime


def _handle_init(state, args):
    """打开共享帧并连接 CARLA。"""
    cfg = build_config(args["config"])
    frame = SharedFrame(
        args["frame"]["name"], args["frame"]["size_bytes"],
        args["frame"]["backing_path"], create=False)
    runtime = CarlaRuntime(cfg, frame)
    state.update({"frame": frame, "runtime": runtime})
    info = runtime.server_info()
    info["python_version"] = [sys.version_info.major, sys.version_info.minor]
    return info


def _handle_query_routes(state, _args):
    return {"spawn_points": state["runtime"].query_spawn_points()}


def _handle_reset(state, args):
    return state["runtime"].reset(int(args["seed"]), args["route"])


def _handle_step(state, args):
    return state["runtime"].step(args["control"])


def _close(state):
    runtime = state.pop("runtime", None)
    if runtime is not None:
        runtime.close()
    frame = state.pop("frame", None)
    if frame is not None:
        frame.close()


_HANDLERS = {
    P.CMD_INIT: _handle_init,
    P.CMD_QUERY_ROUTES: _handle_query_routes,
    P.CMD_RESET: _handle_reset,
    P.CMD_STEP: _handle_step,
}


def main():
    """运行协议循环，直到 shutdown 或父进程关闭 stdin。"""
    protocol_out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    protocol_in = sys.stdin.buffer
    sys.stdout = sys.stderr
    state = {}
    try:
        while True:
            message = read_message(protocol_in)
            if message is None:
                break
            try:
                command = message["cmd"]
                if command == P.CMD_SHUTDOWN:
                    _close(state)
                    write_message(protocol_out, make_response(True))
                    break
                if command not in _HANDLERS:
                    raise ValueError("未知闭环命令: {}".format(command))
                result = _HANDLERS[command](state, message.get("args", {}))
                write_message(protocol_out, make_response(True, result))
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                write_message(protocol_out, make_response(False, error=repr(exc)))
    finally:
        _close(state)


if __name__ == "__main__":
    main()
