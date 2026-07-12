"""采集主循环：建队列→逐路线驱动 worker→分段落盘→从共享内存读帧→编码+写 LMDB。

模块: collector/orchestrator/orchestrator.py
依赖: os, dataclasses, pathlib, numpy, common.*, collector.*
读取配置: carla_collector 全树（ipc/worker/route/weather/cameras/output/collision/simulation 等）
对外接口:
    - run(cfg, max_scenes_override=None) -> int     # 执行采集，返回成功落盘的场景段数
说明: 父进程创建共享内存 arena 并持有，worker 子进程写入。一次行驶随 arena 反复写满被切成多个连续段，
      每段落一个自包含场景目录（共享 drive_id、segment_idx 递增），worker 在多次 RPC 间保活世界续采，
      直到到达终点或达整次行驶总帧上限。碰撞丢弃当前未落盘段、保留已落段、结束行驶；本次行驶 0 段产出
      才换种子重试（Design ④）。读帧时用生成器惰性消费 arena，使内存只驻留一帧（深度解码、lidar 还原
      均在此 Py312 侧做）。RGB→mp4、其余→LMDB（Design ⑧）；具体落哪些模态由 cameras.modalities 与
      lidar.enabled 开关决定（关闭即不读盘、不落盘，RGB 关则无 mp4），光流与深度同法逐相机入 LMDB。
"""

import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from common import protocol as P
from common.shm import Arena
from collector import scenarios
from collector.encode import encode_camera
from collector.routes import build_route_queue
from collector.worker_proc import WorkerProcess
from collector.writer import LmdbWriter, read_scene_route

_REPO_ROOT = Path(__file__).resolve().parents[4]
_LIDAR_DTYPE = np.dtype(P.SEMANTIC_LIDAR_DTYPE)
_DEPTH_MAX_M = 1000.0  # carla 深度相机编码的最大量程（米）


def _resolve(path):
    p = Path(path)
    return p if p.is_absolute() else _REPO_ROOT / p


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
        yield {"meta": {"frame_id": fr["frame_id"], "sim_time": fr["sim_time"],
                        "ego": fr["ego"], "bboxes": fr["bboxes"],
                        "traffic_light_states": fr["traffic_light_states"]},
               "arrays": arrays}


def _estimate_lmdb_bytes(frames, cam_names, height, width, mods, lidar_on):
    """估算写入 LMDB 的字节：按启用模态累加每相机像素开销 + 语义lidar(原 blob) + 每帧元数据余量。"""
    pixels = height * width * len(cam_names)
    # 深度 float32(4) + 语义图 uint8(1) + 光流双通道 float32(8)，按开关计入
    per_pixel = (4 if mods.depth else 0) + (1 if mods.semantic else 0) + (8 if mods.optical_flow else 0)
    per_frame = pixels * per_pixel + 64 * 1024  # 像素开销 + 元数据余量
    lidar_total = sum(fr["blobs"]["lidar"][P.BLOB_SIZE] for fr in frames) if lidar_on else 0
    return per_frame * len(frames) + lidar_total


def _persist(scene_id, route, seed, weather, frames, status, static_meta,
             drive_id, segment_idx, cc, arena, output_root, cam_names):
    """把一段（一次 arena flush）落盘为自包含场景目录：每相机 RGB→mp4，深度/lidar/标注→独立 LMDB。

    同一次行驶切出的多段共享 drive_id、segment_idx 递增，下游据此可拼回完整路线。
    """
    scene_dir = output_root / "scenes" / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    mods = cc.cameras.modalities
    lidar_on = cc.lidar.enabled

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
        "num_frames": len(frames), "map": cc.simulation.map, "fps": cc.output.video_fps,
        "drive_id": drive_id, "segment_idx": segment_idx,
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
        writer.write_scene(scene_meta, _frame_payloads(arena, frames, cam_names, mods, lidar_on),
                           est_bytes=est)
    finally:
        writer.close()


def _collect_route(worker, route, saved, cc, arena, output_root, cam_names, rng, weather_presets):
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
        print("[collector] {} 开始行驶 {} seed={} attempt={}/{}".format(
            next_scene, route_tag, seed, attempt + 1, retries + 1))
        r = worker.start_scene(seed, weather, {"start": route["start"], "end": route["end"]})
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

        while True:
            if status == P.STATUS_COLLISION:
                print("[collector] {}（{}）碰撞，丢弃当前未落盘段，结束行驶".format(drive_id, route_tag))
                break
            if frames:  # partial/ok/max_frames 段均落盘
                scene_id = "scene_{:06d}".format(saved + segs_total)
                _persist(scene_id, route, seed, weather, frames, status, static_meta,
                         drive_id, seg_idx, cc, arena, output_root, cam_names)
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

        if status == P.STATUS_COLLISION and segs_drive == 0:
            if attempt < retries:                       # 还有重试余额，才是真的换种子重试
                print("[collector] {} 首段前碰撞，换种子重试（剩余 {} 次）".format(route_tag, retries - attempt))
                continue                                # 本次行驶 0 段产出 → 复用 scene 编号重试同路线
            print("[collector] {} 首段前碰撞，无重试余额(max_retries={})，跳过".format(route_tag, retries))
        return segs_total  # 有产出 / 到终点 / 重试耗尽 → 本路线完成


def _scan_existing(output_root):
    """断点续采：扫描已存在场景目录，返回 (已采路线键集合, 下一个场景编号)。

    路线键 = (start_idx, end_idx)，取自各场景 LMDB 的 meta；据此把已采过的路线从队列剔除
    （无论该次行驶是否跑完，只要落过盘就排除）。编号取已存在 scene_XXXXXX 的最大序号 +1，
    使本次新段从全新编号续写，绝不覆盖既有数据（含 LMDB 不可读的半成品目录）。
    """
    scenes_dir = output_root / "scenes"
    done_routes = set()
    max_idx = -1
    if not scenes_dir.is_dir():
        return done_routes, 0
    for d in sorted(scenes_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("scene_"):
            continue
        try:
            max_idx = max(max_idx, int(d.name.split("_")[1]))
        except (IndexError, ValueError):
            pass  # 命名不符的目录不参与编号推进
        route_key = read_scene_route(d / "lmdb")
        if route_key is not None:
            done_routes.add(route_key)
    return done_routes, max_idx + 1


def run(cfg, max_scenes_override=None):
    """执行采集主循环，返回成功落盘的场景段数。"""
    cc = cfg.carla_collector
    output_root = _resolve(cc.output.root)
    output_root.mkdir(parents=True, exist_ok=True)
    cam_names = [c.name for c in cc.cameras.rig]
    # 断点续采：识别已采路线与续写起始编号（输出目录非空时生效）
    done_routes, start_index = _scan_existing(output_root)

    arena_name = "{}_{}".format(cc.ipc.arena_name, os.getpid())
    arena_size = cc.ipc.arena_size_mb * 1024 * 1024
    arena = Arena(arena_name, arena_size, create=True)  # 父进程创建并持有，保证区域存活
    worker = WorkerProcess(_resolve(cc.worker.python_exe))
    master_rng = np.random.RandomState()  # 不固定：场景种子真随机，但逐场景记录

    saved = start_index  # 续写编号从已存在场景之后开始，不覆盖既有数据
    try:
        info = worker.init(asdict(cfg), arena_name, arena_size)
        print("[collector] worker 就绪:", info)
        weather_presets = info["weather_presets"]  # 随机天气从 worker 实际拥有的内置预设中选
        spawn_points = worker.query_spawn_points()

        max_scenes = max_scenes_override if max_scenes_override is not None else cc.route.max_scenes
        # 先建全量有序队列再剔除已采路线，最后才按 max_scenes 裁剪：使续采时 max_scenes 表示
        # 「本次再采多少条新路线」，而非被已采路线占满名额
        full_queue = build_route_queue(spawn_points, cc.route.min_distance_m,
                                       cc.route.max_distance_m, cc.route.queue_seed, 0)
        queue = [r for r in full_queue if (r["start_idx"], r["end_idx"]) not in done_routes]
        skipped = len(full_queue) - len(queue)
        if max_scenes:
            queue = queue[:max_scenes]
        if skipped:
            print("[collector] 断点续采：剔除 {} 条已采路线".format(skipped))
        if start_index:
            print("[collector] 断点续采：新场景从 scene_{:06d} 起编号".format(start_index))
        print("[collector] 路线队列长度:", len(queue))

        for route in queue:
            # 一条路线（一次行驶）可能切成多段落盘，saved 据返回段数推进
            saved += _collect_route(worker, route, saved, cc, arena,
                                    output_root, cam_names, master_rng, weather_presets)
        print("[collector] 完成，成功落盘场景段数:", saved)
    finally:
        worker.shutdown()
        arena.close()
    return saved
