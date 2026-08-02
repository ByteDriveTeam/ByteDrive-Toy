"""隔离 Poisson 子进程入口：关闭 Windows 崩溃弹窗并返回单次表面重建结果。

模块: data/mesh_reconstruction/surface/worker.py
依赖: argparse, ctypes, os, pathlib, types, torch,
      data.mesh_reconstruction.surface.surface
读取配置: —（全部参数由项目内 request PT 显式传入）
对外接口:
    - main() -> None
说明: 仅由 reconstruct_surface_isolated 派生，不作为用户 CLI。
"""

import argparse
import ctypes
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from data.mesh_reconstruction.surface.surface import reconstruct_surface

_REPO_ROOT = Path(__file__).resolve().parents[3]


def main():
    """读取一次请求、执行 Poisson 并保存响应。"""
    _disable_windows_error_dialogs()
    os.environ["BYTEDRIVE_MESH_WORKER"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("request")
    parser.add_argument("response")
    args = parser.parse_args()
    request, response = Path(args.request).resolve(), Path(args.response).resolve()
    assert _REPO_ROOT in request.parents and _REPO_ROOT in response.parents, \
        "隔离 Poisson 请求与响应必须位于项目目录内"
    payload = torch.load(request, map_location="cpu", weights_only=True)
    cfg = SimpleNamespace(**payload["cfg"])
    result = reconstruct_surface(
        payload["points"], payload["tags"], payload["orientation_targets"], cfg,
        payload["hole_radius_m"], payload["enable_watertight_repair"],
        torch.device(payload["device"]),
        payload["batch_size"], payload["candidate_budget"], payload["poisson_threads"])
    torch.save(result, response)


def _disable_windows_error_dialogs():
    if os.name == "nt":
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)


if __name__ == "__main__":
    main()
