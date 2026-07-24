"""串联 Py37 CARLA、共享 RGB、驾驶模型、轨迹控制与逐 episode 评测日志。

模块: clone_loop/orchestrator/orchestrator.py
依赖: dataclasses, os, pathlib, select, sys, numpy,
      clone_loop.client/control/inference/logger/protocol/recorder/routes/shared_frame,
      clone_loop.orchestrator.checks.orchestrator_checks
读取配置:
    clone_loop.worker.python_exe
    clone_loop.ipc.frame_name
    clone_loop.simulation.base_seed
    clone_loop.route.min_distance_m / max_distance_m / max_episodes / queue_seed
    clone_loop.camera.width / height
    clone_loop.control.* / clone_loop.simulation.fixed_delta_seconds
    clone_loop.output.root / log_every
    （模型推理、worker 与各子模块继续读取其文件头所列配置）
对外接口:
    - run_closed_loop(cfg, max_episodes_override=None) -> dict
"""

from dataclasses import asdict
import os
from pathlib import Path
import select
import sys

import numpy as np

from clone_loop import protocol as P
from clone_loop.client import WorkerClient
from clone_loop.control import TrajectoryController
from clone_loop.inference import ClosedLoopPolicy
from clone_loop.logger import RunLogger
from clone_loop.orchestrator.checks.orchestrator_checks import (
    check_episode_override,
    check_runtime_versions,
    check_output_root,
    check_routes,
)
from clone_loop.recorder import EpisodeRecorder
from clone_loop.routes import build_route_queue
from clone_loop.shared_frame import SharedFrame


__all__ = ["run_closed_loop"]

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_output(path):
    output = Path(path)
    output = output if output.is_absolute() else _REPO_ROOT / output
    output = output.resolve()
    check_output_root(output, _REPO_ROOT.resolve())
    output.mkdir(parents=True, exist_ok=True)
    return output


def _frame_array(shared, height, width):
    """复制当前共享帧，使下一次 worker 写入与本步 PyTorch 消费完全解耦。"""
    return np.frombuffer(shared.read(), dtype=np.uint8).reshape(height, width, 3).copy()


def _manual_stop_requested():
    """非阻塞读取控制台；Windows 单按 q，其他终端输入 q 后回车。"""
    if sys.stdin is None or not sys.stdin.isatty():
        return False
    if os.name == "nt":
        import msvcrt
        requested = False
        while msvcrt.kbhit():
            requested = msvcrt.getwch().lower() == "q" or requested
        return requested
    readable, _, _ = select.select([sys.stdin], [], [], 0.0)
    return bool(readable and sys.stdin.readline().strip().lower() == "q")


def _episode(worker, shared, policy, controller, recorder, logger, route, seed, cfg):
    """运行一条路线直至 worker 返回终态，并返回 episode 汇总。"""
    policy.reset()
    controller.reset()
    observation = worker.reset(seed, route)
    while observation["status"] == P.STATUS_RUNNING:
        if _manual_stop_requested():
            observation = dict(observation, status=P.STATUS_MANUAL)
            print("[clone_loop] 已手动结束当前 episode")
            break
        frame = _frame_array(shared, cfg.camera.height, cfg.camera.width)
        decision = policy.infer(frame, observation)
        command = controller.command(
            decision["trajectory"], observation["speed_mps"],
            decision["behavior_probabilities"])
        recorder.write(frame, decision, observation, command)
        observation = worker.step(command)
        logger.write_step(observation, command, decision)
        if observation["step"] % cfg.output.log_every == 0:
            print("[clone_loop] step={} progress={:.1%} speed={:.2f}m/s mode={}".format(
                observation["step"], observation["route_completion"],
                observation["speed_mps"], decision["mode"]))
    recorder.write_terminal(_frame_array(
        shared, cfg.camera.height, cfg.camera.width))
    recorder.close()
    return logger.finish_episode(observation, recorder.artifacts)


def run_closed_loop(cfg, max_episodes_override=None):
    """执行配置的 CARLA 闭环路线队列并返回运行级汇总。"""
    check_episode_override(max_episodes_override)
    cl = cfg.clone_loop
    output_root = _resolve_output(cl.output.root)
    frame_size = cl.camera.width * cl.camera.height * 3
    frame_name = "{}_{}".format(cl.ipc.frame_name, os.getpid())
    backing_path = output_root / (frame_name + ".bin")
    shared = SharedFrame(frame_name, frame_size, backing_path, create=True)
    worker = None
    logger = None
    try:
        worker = WorkerClient(cl.worker.python_exe)
        logger = RunLogger(output_root)
        controller = TrajectoryController(cl.control, cl.simulation.fixed_delta_seconds)
        info = worker.init(asdict(cfg), frame_name, frame_size, backing_path)
        check_runtime_versions(info)
        print("[clone_loop] worker 就绪: {}".format(info))
        spawn_points = worker.query_spawn_points()
        limit = cl.route.max_episodes \
            if max_episodes_override is None else max_episodes_override
        routes = build_route_queue(
            spawn_points, cl.route.min_distance_m, cl.route.max_distance_m,
            cl.route.queue_seed, limit)
        check_routes(routes)
        policy = ClosedLoopPolicy(cfg)
        print("[clone_loop] 路线数: {}".format(len(routes)))
        if sys.stdin is not None and sys.stdin.isatty():
            print("[clone_loop] 运行中按 q 可结束当前 episode 并继续下一条路线")
        for index, route in enumerate(routes):
            seed = cl.simulation.base_seed + index
            logger.start_episode(index, route, seed)
            print("[clone_loop] episode={} route={}->{} seed={}".format(
                index, route["start_idx"], route["end_idx"], seed))
            recorder = EpisodeRecorder(logger.run_dir, index, cfg)
            try:
                summary = _episode(
                    worker, shared, policy, controller, recorder, logger, route, seed, cl)
            finally:
                recorder.close()
            print("[clone_loop] episode={} status={} progress={:.1%} steps={}".format(
                index, summary["status"], summary["route_completion"], summary["steps"]))
        aggregate = logger.finish_run()
        aggregate["run_dir"] = str(logger.run_dir)
        print("[clone_loop] 完成：成功 {}/{}，日志 {}".format(
            aggregate["successes"], aggregate["num_episodes"], logger.run_dir))
        return aggregate
    finally:
        if logger is not None:
            logger.close()
        if worker is not None:
            worker.shutdown()
        shared.close()
