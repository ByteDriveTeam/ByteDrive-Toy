"""派生 Py37 CARLA worker，并以同步 JSON RPC 驱动闭环 episode。

模块: clone_loop/client/client.py
依赖: os, subprocess, pathlib, clone_loop.protocol, clone_loop.client.checks.client_checks
读取配置: 由构造与 init 参数接收 clone_loop.worker.python_exe 及共享帧信息
对外接口:
    - WorkerClient(python_exe)
        .init(config_dict, frame_name, frame_size, backing_path) -> dict
        .query_spawn_points() -> list
        .reset(seed, route) -> dict
        .step(control) -> dict
        .shutdown() -> None
"""

import os
import subprocess
from pathlib import Path

from clone_loop import protocol as P
from clone_loop.client.checks.client_checks import check_python_executable, check_response
from clone_loop.protocol import make_command, read_message, write_message


__all__ = ["WorkerClient"]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKER_MAIN = _REPO_ROOT / "clone_loop" / "worker" / "run.py"
_COLLECTOR_ROOT = _REPO_ROOT / "data" / "carla_data_collector"


class WorkerClient:
    """闭环 Py37 worker 的同步 RPC 客户端。"""

    def __init__(self, python_exe):
        executable = Path(python_exe)
        executable = executable if executable.is_absolute() else _REPO_ROOT / executable
        check_python_executable(executable)
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join((
            str(_REPO_ROOT), str(_COLLECTOR_ROOT), environment.get("PYTHONPATH", "")))
        self._process = subprocess.Popen(
            [str(executable), str(_WORKER_MAIN)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            cwd=str(_REPO_ROOT), env=environment)

    def _request(self, command, **args):
        write_message(self._process.stdin, make_command(command, **args))
        response = read_message(self._process.stdout)
        check_response(response, command)
        return response["result"]

    def init(self, config_dict, frame_name, frame_size, backing_path):
        """下发配置并让 worker 打开父进程已创建的共享帧。"""
        return self._request(
            P.CMD_INIT, config=config_dict,
            frame={"name": frame_name, "size_bytes": frame_size,
                   "backing_path": str(backing_path)})

    def query_spawn_points(self):
        """查询当前地图推荐生成点。"""
        return self._request(P.CMD_QUERY_ROUTES)["spawn_points"]

    def reset(self, seed, route):
        """重置世界并取得 episode 首帧观测。"""
        return self._request(P.CMD_RESET, seed=seed, route=route)

    def step(self, control):
        """应用控制并取得下一帧观测。"""
        return self._request(P.CMD_STEP, control=control)

    def shutdown(self):
        """请求 worker 清理退出；协议失败时终止进程兜底。"""
        try:
            if self._process.poll() is None:
                self._request(P.CMD_SHUTDOWN)
                self._process.wait(timeout=15)
        except Exception:
            pass
        if self._process.poll() is None:
            self._process.terminate()
