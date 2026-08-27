"""专家/模型双模式采集主循环：闭环推进、分段落盘与完整 GT 代价回填。

模块: collector/orchestrator/orchestrator.py
依赖: math, os, dataclasses, pathlib, numpy, clone_loop.*, common.*, collector.*
读取配置: carla_collector 全树、clone_loop.ipc/control、data.driving.cameras
对外接口:
    - run(cfg, max_scenes_override=None) -> int     # 执行采集，返回成功落盘的场景段数
说明: behavior_agent 保持原专家采集语义；model 模式只用 Winner 轨迹产生控制，行为概率仅存档。
      父进程创建共享内存 arena 并持有，worker 子进程写入。一次行驶随 arena 反复写满被切成多个连续段，
      每段落一个自包含场景目录（共享 drive_id、segment_idx 递增），worker 在多次 RPC 间保活世界续采，
      直到到达终点或达整次行驶总帧上限。专家碰撞沿用原重试；模型碰撞进入恢复窗口且失败数据也保留。本次行驶 0 段产出
      才换种子重试（Design ④）。读帧时用生成器惰性消费 arena，使内存只驻留一帧（深度解码、lidar 还原
      均在此 Py312 侧做）。RGB→mp4、其余→LMDB（Design ⑧）；具体落哪些模态由 cameras.modalities 与
      lidar.enabled 开关决定（关闭即不读盘、不落盘，RGB 关则无 mp4），光流与深度同法逐相机入 LMDB。
      worker 生成的 CARLA 原生路线相关交通控制点随低频帧元数据落盘；10Hz 运动学写入独立 LMDB 时间轴，
      以 frame_id/sim_time 与传感器帧关联。模型预测与完整未来 10Hz GT 在行驶结束后生成逐候选逐点原始代价；
      不裁剪、不归一化、不加权也不保存总分。字段缺失不补默认值，使下游能识别旧数据。
"""

import math
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from common import protocol as P
from common.shm import Arena
from clone_loop.control import TrajectoryController
from clone_loop.inference import ClosedLoopPolicy
from clone_loop.shared_frame import SharedFrame
from collector import scenarios
from collector.costs import COST_TERMS, evaluate_drive_costs
from collector.encode import encode_camera
from collector.routes import build_route_queue
from collector.worker_proc import WorkerProcess
from collector.writer import (
    LmdbWriter, append_model_data, compact_lmdb, read_scene_identity,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_LIDAR_DTYPE = np.dtype(P.SEMANTIC_LIDAR_DTYPE)
_DEPTH_MAX_M = 1000.0  # carla 深度相机编码的最大量程（米）


def _resolve(path):
    p = Path(path)
    return p if p.is_absolute() else _REPO_ROOT / p


def _resolve_output(path):
    """解析写入根并拒绝项目目录外目标。"""
    output = _resolve(path).resolve()
    if os.path.commonpath((str(_REPO_ROOT.resolve()), str(output))) \
            != str(_REPO_ROOT.resolve()):
        raise ValueError("carla_collector.output.root 必须位于项目目录内")
    return output


def _blob_array(arena, blob):
    """按 blob 描述从 arena 零拷贝读出 ndarray（lidar 用结构化 dtype 还原）。"""
    buf = arena.read(blob[P.BLOB_OFFSET], blob[P.BLOB_SIZE])
    dtype = _LIDAR_DTYPE if blob[P.BLOB_DTYPE] == "semantic_lidar" else np.dtype(blob[P.BLOB_DTYPE])
    return np.frombuffer(buf, dtype=dtype).reshape(blob[P.BLOB_SHAPE])


def _decode_depth(bgr):
    """carla 深度图解码：BGR 三通道编码值 -> 米（float32）。"""
    arr = bgr.astype(np.float32)
    normalized = (arr[..., 2] + arr[..., 1] * 256.0 + arr[..., 0] * 65536.0) / (256.0 ** 3 - 1.0)
    return (_DEPTH_MAX_M * normalized).astype(np.float32)


def _rgb_frames(arena, frames, cam):
    """惰性产出某相机的逐帧 BGR 图（供编码器流式消费，避免整段 RGB 驻留内存）。"""
    key = "rgb/" + cam
    return (_blob_array(arena, fr["blobs"][key]) for fr in frames)


def _frame_payloads(arena, frames, cam_names, mods, lidar_on):
    """惰性产出每帧的 LMDB 负载：按开关取 深度/语义图/光流（每相机）+ 语义 lidar + 小元数据。

    仅 RGB 走 mp4、不入 LMDB；其余启用模态各自落 LMDB。语义图/光流已是最终值，仅拷出 arena 视图
    （避免被下一帧覆盖）；深度需由编码 BGR 解码为米。
    """
    for fr in frames:
        arrays = {}
        if mods.depth:
            arrays.update({"depth/" + cam: _decode_depth(_blob_array(arena, fr["blobs"]["depth/" + cam]))
                           for cam in cam_names})
        if mods.semantic:
            arrays.update({"semantic/" + cam: np.array(_blob_array(arena, fr["blobs"]["semantic/" + cam]))
                           for cam in cam_names})
        if mods.optical_flow:
            arrays.update({"optical_flow/" + cam: np.array(_blob_array(arena, fr["blobs"]["optical_flow/" + cam]))
                           for cam in cam_names})
        if lidar_on:
            arrays["lidar"] = np.array(_blob_array(arena, fr["blobs"]["lidar"]))  # 拷出结构化数组
        metadata = {
            "frame_id": fr["frame_id"], "sim_time": fr["sim_time"],
            "ego": fr["ego"], "bboxes": fr["bboxes"],
            "traffic_light_states": fr["traffic_light_states"],
        }
        if "relevant_traffic_control" in fr:
            metadata["relevant_traffic_control"] = fr["relevant_traffic_control"]
        yield {"meta": metadata, "arrays": arrays}


def _estimate_lmdb_bytes(frames, cam_names, height, width, mods, lidar_on):
    """估算写入 LMDB 的字节：按启用模态累加每相机像素开销 + 语义lidar(原 blob) + 每帧元数据余量。"""
    pixels = height * width * len(cam_names)
    # 深度 float32(4) + 语义图 uint8(1) + 光流双通道 float32(8)，按开关计入
    per_pixel = (4 if mods.depth else 0) + (1 if mods.semantic else 0) + (8 if mods.optical_flow else 0)
    per_frame = pixels * per_pixel + 64 * 1024  # 像素开销 + 元数据余量
    lidar_total = sum(fr["blobs"]["lidar"][P.BLOB_SIZE] for fr in frames) if lidar_on else 0
    return per_frame * len(frames) + lidar_total


def _persist(scene_id, map_name, route, seed, weather, frames, kinematics, status, static_meta,
             drive_id, segment_idx, cc, arena, output_root, cam_names,
             controller="behavior_agent", complete=True, compact=True):
    """把一段落盘：RGB→mp4，其他传感器/标注与独立运动学时间轴→LMDB。

    同一次行驶切出的多段共享 drive_id、segment_idx 递增，下游据此可拼回完整路线。
    """
    scene_dir = output_root / "scenes" / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    mods = cc.cameras.modalities
    lidar_on = cc.lidar.enabled
    if cc.simulation.no_rendering_mode:
        # worker 在无渲染模式不创建视觉传感器，落盘侧也必须使用同一组有效模态。
        mods = type(mods)(rgb=False, depth=False, semantic=False, optical_flow=False)
        lidar_on = False

    # RGB 关闭则本场景无 mp4；其余模态各自落 LMDB
    video_files = {}
    if mods.rgb:
        for cam in cam_names:
            out = scene_dir / "rgb_{}.mp4".format(cam)
            encode_camera(_rgb_frames(arena, frames, cam), out, cc.output.video_codec,
                          cc.output.video_crf, cc.output.video_fps, cc.cameras.width, cc.cameras.height)
            video_files[cam] = out.name  # 相对场景目录，单场景自描述

    scene_meta = {
        "scene_id": scene_id, "seed": seed, "weather": weather, "status": status,
        "num_frames": len(frames), "num_kinematics": len(kinematics),
        "map": map_name, "fps": cc.output.video_fps,
        "sensor_dt_s": cc.simulation.fixed_delta_seconds * cc.collection.capture_every_n_ticks,
        "kinematics_dt_s": (
            cc.simulation.fixed_delta_seconds * cc.collection.kinematics_every_n_ticks),
        "drive_id": drive_id, "segment_idx": segment_idx,
        "controller": controller, "complete": bool(complete),
        "route": {k: route[k] for k in ("start_idx", "end_idx", "start", "end")},
        "intrinsics": static_meta["intrinsics"], "extrinsics": static_meta["extrinsics"],
        "lidar_extrinsic": static_meta["lidar_extrinsic"], "static_bboxes": static_meta["static_bboxes"],
        "traffic_lights": static_meta["traffic_lights"],
        "camera_names": cam_names, "video_files": video_files,
    }
    est = _estimate_lmdb_bytes(frames, cam_names, cc.cameras.height, cc.cameras.width, mods, lidar_on)
    # 每段一个独立 LMDB，co-located 于场景目录；map_size 上限为单段量级
    writer = LmdbWriter(scene_dir / "lmdb", cc.output.lmdb_map_size_gb)
    try:
        writer.write_scene(
            scene_meta, _frame_payloads(arena, frames, cam_names, mods, lidar_on),
            kinematics=kinematics, est_bytes=est)
    finally:
        writer.close()
    if compact:
        compact_lmdb(scene_dir / "lmdb", verify=True)
    return scene_dir


def _collect_route(worker, map_name, route, saved, cc, arena, output_root, cam_names, rng,
                   weather_presets):
    """采集单条路线：一次行驶随 arena 反复写满被切成多段，逐段落盘。返回本路线落盘的段数。

    每填满一次 arena（partial）→ 落一段、reset、续采；到终点(ok)/达总帧上限(max_frames) → 末段落盘、行驶结束。
    碰撞 → 丢弃当前未落盘段、保留已落段、结束行驶；若本次行驶 0 段产出则换种子重试同路线（Design ④）。
    同一次行驶的各段共享 drive_id（= 首段 scene_id），segment_idx 从 0 递增。
    """
    segs_total = 0  # 本路线累计落盘段数（用于推进全局 scene 编号）
    # 路线标识：带 spawn 索引与起终点平面坐标，用于在日志里肉眼判断「是否同一条路线/同一地点」
    s, e = route["start"], route["end"]
    route_tag = "路线 {}->{} 起[{:.0f},{:.0f}]→终[{:.0f},{:.0f}]".format(
        route["start_idx"], route["end_idx"], s[0], s[1], e[0], e[1])
    retries = cc.collision.max_retries_per_route
    for attempt in range(retries + 1):
        seed = scenarios.random_seed(rng)
        weather = scenarios.random_weather(rng, cc.weather.randomize, weather_presets)
        next_scene = "scene_{:06d}".format(saved + segs_total)
        # 每次行驶起始即打印「本场景跑的是哪条路线 + 种子 + 第几次尝试」，据此判断是否在重复同一路线
        print("[collector] {} 开始行驶 map={} {} seed={} attempt={}/{}".format(
            next_scene, map_name, route_tag, seed, attempt + 1, retries + 1))
        r = worker.start_scene(
            map_name, seed, weather, {"start": route["start"], "end": route["end"]})
        status = r["status"]
        if status == P.STATUS_UNREACHABLE:
            print("[collector] {} 不可达，跳过".format(route_tag))
            return segs_total

        # 内外参/静态框整次行驶不变，仅首段回传，供本次行驶所有段复用
        static_meta = {k: r[k] for k in
                       ("intrinsics", "extrinsics", "lidar_extrinsic", "static_bboxes",
                        "traffic_lights")}
        drive_id = "scene_{:06d}".format(saved + segs_total)
        seg_idx = 0
        segs_drive = 0  # 本次行驶（本 attempt）已落段数
        frames = r["frames"]
        kinematics = r["kinematics"]

        while True:
            if status == P.STATUS_COLLISION:
                print("[collector] {}（{}）碰撞，丢弃当前未落盘段，结束行驶".format(drive_id, route_tag))
                break
            if frames:  # partial/ok/max_frames 段均落盘
                scene_id = "scene_{:06d}".format(saved + segs_total)
                _persist(scene_id, map_name, route, seed, weather, frames, kinematics, status,
                         static_meta, drive_id, seg_idx, cc, arena, output_root, cam_names)
                print("[collector] {} 落盘段 #{}（{}帧, status={}）".format(
                    scene_id, seg_idx, len(frames), status))
                segs_total += 1
                seg_idx += 1
                segs_drive += 1
            if status in (P.STATUS_OK, P.STATUS_MAX_FRAMES):
                break  # 行驶完成
            # status == PARTIAL：arena 已被本进程读空，命 worker reset 后续采下一段
            if not frames:
                # partial 却 0 帧 = arena 装不下单帧，否则会无限续采空段
                raise RuntimeError(
                    "arena 容量不足以容纳单帧（ipc.arena_size_mb={} 太小），无法续采".format(
                        cc.ipc.arena_size_mb))
            r = worker.continue_scene()
            status = r["status"]
            frames = r["frames"]
            kinematics = r["kinematics"]

        if status == P.STATUS_COLLISION and segs_drive == 0:
            if attempt < retries:                       # 还有重试余额，才是真的换种子重试
                print("[collector] {} 首段前碰撞，换种子重试（剩余 {} 次）".format(route_tag, retries - attempt))
                continue                                # 本次行驶 0 段产出 → 复用 scene 编号重试同路线
            print("[collector] {} 首段前碰撞，无重试余额(max_retries={})，跳过".format(route_tag, retries))
        return segs_total  # 有产出 / 到终点 / 重试耗尽 → 本路线完成


def _live_frame_array(shared, views, height, width):
    """复制当前模型 RGB 共享帧，避免下一条 worker 命令覆盖。"""
    return np.frombuffer(shared.read(), dtype=np.uint8).reshape(
        views, height, width, 3).copy()


def _live_lidar_array(shared, count):
    """按 worker 回传点数复制当前模型 XYZ LiDAR。"""
    size = int(count) * 3 * np.dtype(np.float32).itemsize
    return np.frombuffer(shared.read_prefix(size), dtype=np.float32).reshape(
        int(count), 3).copy()


def _model_step(decision, command, current_state, next_state, waypoint_dt_s,
                world_index):
    """构造可落盘模型记录；行为概率仅记录，不进入控制。"""
    return {
        "input_frame_id": int(current_state["frame_id"]),
        "next_frame_id": int(next_state["frame_id"]),
        "world_index": int(world_index),
        "winner_mode": int(decision["mode"]),
        "history_valid": bool(decision["history_valid"]),
        "waypoint_dt_s": float(waypoint_dt_s),
        "control_source": "winner_trajectory_only",
        "control": {key: float(value) for key, value in command.items()},
        "confidence": np.asarray(decision["confidence"]).tolist(),
        "mode_scores": np.asarray(decision["mode_scores"]).tolist(),
        "behavior_probabilities": np.asarray(
            decision["behavior_probabilities"]).tolist(),
        "trajectories": np.asarray(decision["trajectories"], dtype=np.float32),
    }


def _persist_model_segment(saved, record_index, map_name, route, seed, weather,
                           status, static_meta, drive_id, cc, arena, output_root,
                           cam_names, segment):
    """落一个尚未代价回填的模型段，并返回待最终化记录。"""
    scene_id = "scene_{:06d}".format(saved + record_index)
    kinematics = [{"frame_id": state["frame_id"], "sim_time": state["sim_time"],
                   "ego": state["ego"]} for state in segment["world_states"]]
    scene_dir = _persist(
        scene_id, map_name, route, seed, weather, segment["frames"], kinematics,
        status, static_meta, drive_id, record_index, cc, arena, output_root,
        cam_names, controller="model", complete=False, compact=False)
    return {
        "scene_dir": scene_dir,
        "world_states": list(segment["world_states"]),
        "model_steps": list(segment["model_steps"]),
    }


def _finalize_model_segments(records, terminal_status, route_geometry, all_world,
                             all_steps, cc, cost_device):
    """用完整未来 GT 统一计算代价，再按段回填和安全压实。"""
    if all_steps:
        evaluate_drive_costs(
            all_world, all_steps, route_geometry, cc.cost, cc.model_collection,
            device=cost_device)
    cost_meta = {
        "complete": True,
        "drive_status": terminal_status,
        "world_state_dt_s": (
            cc.simulation.fixed_delta_seconds
            * cc.collection.world_state_every_n_ticks),
        "model_dt_s": (
            cc.simulation.fixed_delta_seconds
            * cc.collection.model_every_n_ticks),
        "cost_terms": list(COST_TERMS),
        "cost_scopes": {
            "current": "prediction_time_actual_ego_state",
            "candidate": "all_candidates_per_waypoint",
            "next": "actual_state_after_winner_control_one_tick",
            "historical": "raw_per_term_sum_of_actual_next_states",
        },
        "cost_semantics": {
            "direction": "lower_is_better",
            "minimum": 0.0,
            "maximum": None,
            "clipped": False,
            "normalized": False,
            "weighted": False,
            "aggregate_total_stored": False,
            "candidate_environment": "future_10hz_ground_truth_actor_boxes",
            "candidate_light_state": "held_at_prediction_time",
        },
    }
    for index, record in enumerate(records):
        segment_meta = dict(cost_meta)
        segment_meta.update({
            "num_segments": len(records),
            "drive_terminal_segment": index == len(records) - 1,
        })
        append_model_data(
            record["scene_dir"] / "lmdb", cc.output.lmdb_map_size_gb,
            segment_meta, record["world_states"], record["model_steps"])
        compact_lmdb(record["scene_dir"] / "lmdb", verify=True)


def _collect_model_route(worker, shared, shared_lidar, policy, controller, map_name,
                         route, saved, cc, arena, output_root, cam_names, rng,
                         weather_presets, cfg):
    """采集一条模型闭环路线，所有失败状态均保留并完成离线代价回填。"""
    seed = scenarios.random_seed(rng)
    weather = scenarios.random_weather(
        rng, cc.weather.randomize, weather_presets)
    drive_id = "scene_{:06d}".format(saved)
    print("[collector] {} 模型闭环 map={} 路线 {}->{} seed={}".format(
        drive_id, map_name, route["start_idx"], route["end_idx"], seed))
    policy.reset()
    controller.reset()
    response = worker.start_model_scene(
        map_name, seed, weather,
        {"start": route["start"], "end": route["end"]})
    if response["status"] == P.STATUS_UNREACHABLE:
        print("[collector] 模型闭环路线 {}->{} 不可达，跳过".format(
            route["start_idx"], route["end_idx"]))
        return 0
    static_meta = response["static_meta"]
    route_geometry = response["route_geometry"]
    observation = response["observation"]
    current_state = response["world_state"]
    all_world = [current_state]
    all_steps = []
    segment = {
        "frames": [response["sensor_frame"]],
        "world_states": [current_state],
        "model_steps": [],
    }
    records = []
    status = response["status"]
    views = len(cfg.data.driving.cameras)
    while status == P.STATUS_RUNNING:
        frame = _live_frame_array(
            shared, views, cc.cameras.height, cc.cameras.width)
        lidar = _live_lidar_array(shared_lidar, observation["lidar_count"])
        decision = policy.infer(frame, lidar, observation)
        # 唯一控制来源：Winner 轨迹。behavior_probabilities 只进入上面的记录字段。
        command = controller.command(decision["trajectory"], observation["speed_mps"])
        result = worker.model_step(command)
        next_state = result["world_state"]
        step = _model_step(
            decision, command, current_state, next_state,
            cfg.clone_loop.control.waypoint_dt_s, len(all_world) - 1)
        all_steps.append(step)
        segment["model_steps"].append(step)
        all_world.append(next_state)
        status = result["status"]

        if result["pending_capture"]:
            # 边界状态既是上一预测的 next，也承载下一段首个 2Hz 帧；两段各存一份以保持自包含。
            segment["world_states"].append(next_state)
            records.append(_persist_model_segment(
                saved, len(records), map_name, route, seed, weather,
                P.STATUS_PARTIAL, static_meta, drive_id, cc, arena, output_root,
                cam_names, segment))
            flushed = worker.flush_model_segment()
            segment = {
                "frames": [flushed["sensor_frame"]],
                "world_states": [next_state],
                "model_steps": [],
            }
        else:
            segment["world_states"].append(next_state)
            if result["sensor_frame"] is not None:
                segment["frames"].append(result["sensor_frame"])
        observation, current_state = result["observation"], next_state

    if segment["frames"]:
        records.append(_persist_model_segment(
            saved, len(records), map_name, route, seed, weather, status,
            static_meta, drive_id, cc, arena, output_root, cam_names, segment))
    _finalize_model_segments(
        records, status, route_geometry, all_world, all_steps, cc,
        cfg.clone_loop.inference.device)
    print("[collector] {} 模型闭环完成 status={}，落盘 {} 段".format(
        drive_id, status, len(records)))
    return len(records)


def _scan_existing(output_root):
    """断点续采：按地图和控制器扫描已采路线，返回下一个场景编号。

    路线按 meta.map 分组，组内键为 (start_idx, end_idx)；据此只从同一地图的队列剔除已采路线
    （无论该次行驶是否跑完，只要落过盘就排除）。编号取已存在 scene_XXXXXX 的最大序号 +1，
    使本次新段从全新编号续写，绝不覆盖既有数据（含 LMDB 不可读的半成品目录）。
    """
    scenes_dir = output_root / "scenes"
    done_routes_by_map = {}
    max_idx = -1
    if not scenes_dir.is_dir():
        return done_routes_by_map, 0
    for d in sorted(scenes_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("scene_"):
            continue
        try:
            max_idx = max(max_idx, int(d.name.split("_")[1]))
        except (IndexError, ValueError):
            pass  # 命名不符的目录不参与编号推进
        identity = read_scene_identity(d / "lmdb")
        if identity is not None:
            map_name, controller, start_idx, end_idx = identity
            done_routes_by_map.setdefault((map_name, controller), set()).add(
                (start_idx, end_idx))
    return done_routes_by_map, max_idx + 1


def run(cfg, max_scenes_override=None):
    """逐地图执行采集主循环，返回落盘场景段的全局续写编号。

    ``simulation.maps`` 的值分别限制每张地图本次建立的路线队列长度；命令行覆盖值非空时
    统一覆盖每张地图的限制。
    """
    if max_scenes_override is not None and max_scenes_override < 0:
        raise ValueError("max_scenes_override 必须 >= 0（0 表示遍历每张地图的全部路线）")
    cc = cfg.carla_collector
    output_root = _resolve_output(cc.output.root)
    output_root.mkdir(parents=True, exist_ok=True)
    cam_names = [c.name for c in cc.cameras.rig]
    # 断点续采：识别已采路线与续写起始编号（输出目录非空时生效）
    done_routes_by_map, start_index = _scan_existing(output_root)

    arena_name = "{}_{}".format(cc.ipc.arena_name, os.getpid())
    arena_size = cc.ipc.arena_size_mb * 1024 * 1024
    arena = Arena(arena_name, arena_size, create=True)  # 父进程创建并持有，保证区域存活
    worker = WorkerProcess(_resolve(cc.worker.python_exe))
    master_rng = np.random.RandomState()  # 不固定：场景种子真随机，但逐场景记录
    shared = None
    shared_lidar = None
    live = None
    policy = None
    controller = None
    if cc.ego.controller == "model":
        views = len(cfg.data.driving.cameras)
        frame_size = views * cc.cameras.width * cc.cameras.height * 3
        lidar_capacity = max(int(math.ceil(
            cc.lidar.points_per_second * cc.simulation.fixed_delta_seconds)), 1)
        lidar_size = lidar_capacity * 3 * np.dtype(np.float32).itemsize
        live_name = "{}_collector_{}".format(cfg.clone_loop.ipc.frame_name, os.getpid())
        lidar_name = live_name + "_lidar"
        frame_path = output_root / (live_name + ".bin")
        lidar_path = output_root / (lidar_name + ".bin")
        shared = SharedFrame(live_name, frame_size, frame_path, create=True)
        shared_lidar = SharedFrame(
            lidar_name, lidar_size, lidar_path, create=True)
        live = {
            "frame": {"name": live_name, "size_bytes": frame_size,
                      "backing_path": str(frame_path)},
            "lidar": {"name": lidar_name, "size_bytes": lidar_size,
                      "backing_path": str(lidar_path)},
        }
        policy = ClosedLoopPolicy(cfg)
        controller = TrajectoryController(
            cfg.clone_loop.control, cc.simulation.fixed_delta_seconds)

    saved = start_index  # 续写编号从已存在场景之后开始，不覆盖既有数据
    try:
        info = worker.init(asdict(cfg), arena_name, arena_size, live=live)
        print("[collector] worker 就绪:", info)
        weather_presets = info["weather_presets"]  # 随机天气从 worker 实际拥有的内置预设中选
        if start_index:
            print("[collector] 断点续采：新场景从 scene_{:06d} 起编号".format(start_index))

        for map_name, configured_scenes in cc.simulation.maps.items():
            max_scenes = (
                max_scenes_override
                if max_scenes_override is not None
                else configured_scenes
            )
            done_routes = done_routes_by_map.get(
                (map_name, cc.ego.controller), set())
            spawn_points = worker.query_spawn_points(map_name)
            # 已采路线作为优先代表参与相似过滤，避免同地图旧数据中的相邻路线在续采时
            # 换一个端点组合再次入队。场景数最后裁剪，表示「本次在该地图再采多少条新路线」。
            queue = build_route_queue(
                spawn_points, cc.route.min_distance_m, cc.route.max_distance_m,
                cc.route.queue_seed, 0, similarity_threshold=cc.route.similarity_threshold_m,
                excluded_pairs=done_routes,
            )
            if max_scenes:
                queue = queue[:max_scenes]
            print("[collector] 地图 {}：路线队列长度 {}（配置场景数={}）".format(
                map_name, len(queue), max_scenes))
            if done_routes:
                print("[collector] 地图 {} 断点续采：按 {} 条已采路线剔除重复或相似候选".format(
                    map_name, len(done_routes)))

            for route in queue:
                # 一条路线（一次行驶）可能切成多段落盘，saved 据返回段数推进
                if cc.ego.controller == "model":
                    saved += _collect_model_route(
                        worker, shared, shared_lidar, policy, controller,
                        map_name, route, saved, cc, arena, output_root, cam_names,
                        master_rng, weather_presets, cfg)
                else:
                    saved += _collect_route(
                        worker, map_name, route, saved, cc, arena, output_root,
                        cam_names, master_rng, weather_presets)
        print("[collector] 完成，成功落盘场景段数:", saved)
    finally:
        worker.shutdown()
        arena.close()
        if shared is not None:
            shared.close()
        if shared_lidar is not None:
            shared_lidar.close()
    return saved
