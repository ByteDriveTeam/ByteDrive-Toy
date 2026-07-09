"""派生并驱动 Py37 worker 子进程的控制管道客户端。

模块: collector/worker_proc.py
依赖: os, subprocess, common.protocol, common.protocol_checks, collector.worker_proc_checks
读取配置: —（python_exe 等由 orchestrator 解析后传入）
对外接口:
    - WorkerProcess(python_exe)
        .init(config_dict, arena_name, arena_size_bytes) -> dict
        .query_spawn_points() -> list[list[6]]
        .start_scene(seed, weather, route) -> dict     # 一次行驶首段（含内外参/静态框）
        .continue_scene() -> dict                       # 复用存活世界续采下一段
        .shutdown() -> None
        .close() -> None
说明: 用 py37_venv 解释器以子进程方式拉起 worker/main.py，控制走二进制 stdin/stdout 的 JSON 行协议，
      worker 的 stderr 直通父进程终端便于排查。每条命令同步等待一行响应；worker 报错经响应 error 抛出。
"""

import os
import subprocess
from pathlib import Path

from common import protocol as P
from common.protocol import make_command, read_message, write_message
from common.protocol_checks import check_response
from collector.worker_proc_checks import check_python_exe

_MODULE_ROOT = Path(__file__).resolve().parents[1]  # data/carla_data_collector
_REPO_ROOT = _MODULE_ROOT.parents[1]                # 仓库根
_WORKER_MAIN = _MODULE_ROOT / "worker" / "main.py"


class WorkerProcess:
    def __init__(self, python_exe):
        check_python_exe(python_exe)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(_MODULE_ROOT), str(_REPO_ROOT), env.get("PYTHONPATH", "")])
        # stderr=None：worker 日志/栈直通父终端。用默认缓冲：readline 高效（整场景帧索引可能是数 MB
        # 的单行 JSON，逐字节读会极慢），写入端由 write_message 显式 flush 保证命令不滞留。
        self._proc = subprocess.Popen(
            [str(python_exe), str(_WORKER_MAIN)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            cwd=str(_REPO_ROOT), env=env)

    def _request(self, cmd, **args):
        write_message(self._proc.stdin, make_command(cmd, **args))
        resp = read_message(self._proc.stdout)
        if resp is None:
            raise RuntimeError("worker 子进程在响应前退出（stdout EOF），请查看其 stderr 输出")
        check_response(resp)
        if not resp["ok"]:
            raise RuntimeError("worker 执行 {} 失败: {}".format(cmd, resp["error"]))
        return resp["result"]

    def init(self, config_dict, arena_name, arena_size_bytes):
        return self._request(P.CMD_INIT, config=config_dict,
                             arena={"name": arena_name, "size_bytes": arena_size_bytes})

    def query_spawn_points(self):
        return self._request(P.CMD_QUERY_SPAWN_POINTS)["spawn_points"]

    def start_scene(self, seed, weather, route):
        return self._request(P.CMD_START_SCENE, seed=seed, weather=weather, route=route)

    def continue_scene(self):
        return self._request(P.CMD_CONTINUE_SCENE)

    def shutdown(self):
        """请求 worker 自行收尾退出；失败则不阻塞，交给 close() 兜底。"""
        try:
            self._request(P.CMD_SHUTDOWN)
            self._proc.wait(timeout=15)
        except Exception:
            pass
        self.close()

    def close(self):
        if self._proc.poll() is None:
            self._proc.terminate()
