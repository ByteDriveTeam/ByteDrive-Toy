"""Py37 worker 子进程入口：经 stdin/stdout JSON 行协议受 collector 驱动采集。

模块: worker/main.py
依赖: carla, numpy, config.schema, common.protocol, common.shm, worker.*
读取配置: 不读 config 文件；配置由 collector 经 init 命令下发（遵守「config 单一来源」）
对外接口:
    - main() -> None     # 进程入口；循环处理 init / query_spawn_points / start_scene / continue_scene / shutdown
说明: 只在 py37_venv 下运行。stdout 仅承载协议消息，故启动即把 print 重定向到 stderr，
      避免 carla/agents 的零散打印污染消息流。大块帧数据写入共享内存 arena，仅帧索引经协议回传。
"""

import os
import sys
import traceback
from pathlib import Path

# 引导导入根：模块根（agents/common/worker）与仓库根（config）
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))  # data/carla_data_collector
sys.path.insert(0, str(_HERE.parents[3]))  # 仓库根 F:\ByteDrive

import random

import carla
import numpy as np

from config.schema import build_config
from common import protocol as P
from common.protocol import read_message, write_message, make_response
from common.protocol_checks import check_command
from common.shm import Arena, BumpAllocator
from worker import actors, collect, session
from worker.geometry import compute_intrinsics
from worker.main_checks import check_init_args
from worker.sensors import SensorRig


def _handle_init(state, args):
    """连接 Carla、打开共享内存 arena、暂存配置。"""
    check_init_args(args)
    cfg = build_config(args["config"])
    cc = cfg.carla_collector
    state["cc"] = cc
    state["client"] = session.connect(cc.worker.carla_host, cc.worker.carla_port,
                                       cc.worker.startup_timeout_s)
    arena = Arena(args["arena"]["name"], args["arena"]["size_bytes"], create=False)
    state["allocator"] = BumpAllocator(arena)
    # 回传内置天气预设清单：随机决策在 collector 侧，但可选项以本机 CARLA 实际拥有的为准
    return {"carla_version": state["client"].get_server_version(),
            "weather_presets": session.list_weather_presets()}


def _handle_query_spawn_points(state, _args):
    """加载地图并返回全部可达点（collector 据此建路线队列）。"""
    return {"spawn_points": session.query_spawn_points(state["client"], state["cc"].simulation.map)}


def _destroy_actors(client, world, ego, vehicle_ids, crowd, rig):
    """销毁本次行驶创建的传感器与全部 actor。"""
    if rig is not None:
        rig.destroy()
    walker_ids = crowd.walker_ids if crowd is not None else []
    controller_ids = crowd.controller_ids if crowd is not None else []
    actors.destroy_scene_actors(client, world, ego, vehicle_ids, walker_ids, controller_ids)


def _cleanup_drive(state):
    """终态/退出时销毁存活行驶并清除 state["drive"]（幂等：无存活行驶则不做事）。"""
    drive = state.pop("drive", None)
    if drive is not None:
        _destroy_actors(state["client"], drive["world"], drive["ego"],
                        drive["vehicle_ids"], drive["crowd"], drive["rig"])


def _run_chunk(state):
    """跑一段：partial 保活世界供续采，否则销毁收尾。返回 (status, frames)。"""
    cc = state["cc"]
    drive = state["drive"]
    try:
        status, frames = collect.collect_chunk(
            drive["world"], drive["ego"], drive["agent"], drive["rig"], drive["crowd"],
            drive["traffic_lights"], state["allocator"], cc.collection,
            cc.worker.command_timeout_s, drive["counters"])
    except Exception:
        _cleanup_drive(state)  # 采集中途异常也要销毁 actor，避免泄漏
        raise
    if status != P.STATUS_PARTIAL:
        _cleanup_drive(state)
    return status, frames


def _handle_start_scene(state, args):
    """重载地图→布景→预热→采首段；partial 时保活世界供 continue_scene 续采。"""
    cc = state["cc"]
    client = state["client"]
    seed = int(args["seed"])
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))  # 让蓝图/颜色等随机选择也随 seed 复现

    world, tm = session.load_scene_world(
        client, cc.simulation.map, cc.simulation.fixed_delta_seconds, cc.traffic.tm_port, seed)
    state["world"], state["tm"] = world, tm
    session.apply_weather(world, args["weather"])

    ego = None
    vehicle_ids = []
    crowd = None
    rig = None
    try:
        ego, agent = actors.spawn_ego(world, cc.ego.vehicle_filter, cc.ego.behavior,
                                      args["route"]["start"])
        vehicle_ids = actors.spawn_traffic_vehicles(
            client, world, tm, cc.traffic.num_vehicles, cc.traffic.vehicle_filter)
        crowd = actors.spawn_walkers(
            client, world, cc.traffic.num_walkers, cc.traffic.walker_filter,
            cc.traffic.walker_running_pct, cc.traffic.walker_arrival_radius_m)
        rig = SensorRig(world, ego, cc.cameras, cc.lidar)
        state["allocator"].reset()

        end = args["route"]["end"]
        destination = carla.Location(x=end[0], y=end[1], z=end[2])
        reachable, static_bboxes, traffic_lights, traffic_light_metadata = collect.prepare_drive(
            world, agent, cc.simulation, destination)
        if not reachable:
            _destroy_actors(client, world, ego, vehicle_ids, crowd, rig)
            return {"status": P.STATUS_UNREACHABLE, "num_frames": 0, "frames": []}

        # 保活整次行驶所需的全部句柄；intrinsics/extrinsics/static 整次行驶不变，仅首段回传
        state["drive"] = {
            "world": world, "tm": tm, "ego": ego, "agent": agent, "rig": rig,
            "crowd": crowd, "vehicle_ids": vehicle_ids, "static_bboxes": static_bboxes,
            "traffic_lights": traffic_lights, "traffic_light_metadata": traffic_light_metadata,
            "intrinsics": {
                camera.name: compute_intrinsics(cc.cameras.width, cc.cameras.height, camera.fov)
                for camera in cc.cameras.rig
            },
            "extrinsics": rig.extrinsics, "lidar_extrinsic": rig.lidar_extrinsic,
            "counters": {"total": 0, "tick_idx": 0},
        }
    except Exception:
        _destroy_actors(client, world, ego, vehicle_ids, crowd, rig)
        state.pop("drive", None)
        raise

    drive = state["drive"]
    static_meta = {
        "static_bboxes": drive["static_bboxes"], "intrinsics": drive["intrinsics"],
        "extrinsics": drive["extrinsics"], "lidar_extrinsic": drive["lidar_extrinsic"],
        "traffic_lights": drive["traffic_light_metadata"],
    }
    status, frames = _run_chunk(state)  # 可能就地清除 state["drive"]（非 partial）
    return dict(status=status, num_frames=len(frames), frames=frames,
                used_bytes=state["allocator"].used, **static_meta)


def _handle_continue_scene(state, _args):
    """复用存活世界续采下一段。此刻 collector 已读完上段 arena，reset 后重填。"""
    assert state.get("drive") is not None, "continue_scene 前必须有存活的 start_scene 行驶"
    state["allocator"].reset()
    status, frames = _run_chunk(state)
    return {"status": status, "num_frames": len(frames), "frames": frames,
            "used_bytes": state["allocator"].used}


def _restore_async(state):
    """退出前把世界与 TM 恢复异步，避免把 Carla 服务端留在同步等待态。"""
    world = state.get("world")
    if world is not None:
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
    if state.get("tm") is not None:
        state["tm"].set_synchronous_mode(False)


_HANDLERS = {
    P.CMD_INIT: _handle_init,
    P.CMD_QUERY_SPAWN_POINTS: _handle_query_spawn_points,
    P.CMD_START_SCENE: _handle_start_scene,
    P.CMD_CONTINUE_SCENE: _handle_continue_scene,
}


def main():
    # stdout 专供协议。Python 层 sys.stdout 重定向管不到 C 扩展（carla/libcarla）直接写 OS 层
    # fd 1 的零散输出——那类输出会混进消息流，令 collector 读到非 JSON 行而崩溃（长跑偶发）。
    # 故先 dup 出真正的管道独占给协议，再把 fd 1 整体指向 stderr：C 层直写 stdout 的内容落到 stderr
    # 当无害日志，污染不到协议。
    out = os.fdopen(os.dup(1), "wb")  # 协议独占真正的 stdout 管道（BufferedWriter，write_message 显式 flush）
    os.dup2(2, 1)                      # fd 1 -> stderr：C 层直写 stdout 的输出变为无害日志
    inp = sys.stdin.buffer
    sys.stdout = sys.stderr            # Python 层 print 也走 stderr

    state = {}
    while True:
        msg = read_message(inp)
        if msg is None:
            break
        try:
            check_command(msg)
            cmd = msg["cmd"]
            if cmd == P.CMD_SHUTDOWN:
                _cleanup_drive(state)  # collector 在 partial 中途退出时兜底销毁残留 actor
                _restore_async(state)
                write_message(out, make_response(True))
                break
            write_message(out, make_response(True, _HANDLERS[cmd](state, msg["args"])))
        except Exception as exc:  # 把异常回传 collector，同时打印栈到 stderr 便于排查
            traceback.print_exc(file=sys.stderr)
            write_message(out, make_response(False, error=repr(exc)))


if __name__ == "__main__":
    main()
