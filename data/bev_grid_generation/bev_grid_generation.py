"""把 CARLA 全量特权状态离线栅格化为 10 图层位压缩 LMDB。

模块: data/bev_grid_generation/bev_grid_generation.py
依赖: cv2, lmdb, msgpack, numpy, zlib, config.schema.Config, data.hd_map,
      data.driving_targets, vis.data_vis.geometry, 本模块 checks
读取配置:
    model.world_model.grid.front_m / rear_m / left_m / right_m / cell_size_m / layer_names
    data.bev_grid_generation.scene_root / output_dir / map_dir / map_name_template /
        lane_half_width_m / lane_line_width_m / stop_line_width_m / lane_type_to_layer /
        agent_semantics / max_scenes / num_workers / lmdb_map_size_gb / compress_level
对外接口:
    - generate_bev_grids(cfg, scene_limit=None) -> dict
    - read_grid_scene_meta(scene_dir) -> dict
说明: 每帧先 packbits 再 zlib，二值 10×256×256 的上界由 640 KiB 降为 80 KiB；
      地图与动态 Box 均在 CPU 向量化，场景级可并行。灯态绑定每盏灯的全部 CARLA 原生停止线，
      不按自车路线筛选；动态 Agent 只区分车辆和行人且不做相关性过滤。
"""

from __future__ import annotations

import os
import shutil
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import lmdb
import msgpack
import numpy as np

from config.schema import Config
from data.bev_grid_generation.checks.bev_grid_generation_checks import (
    check_generation_paths,
    check_source_scene,
)
from data.driving_targets import BevParams, ego_xy_to_pixel
from data.hd_map import HdMap
from vis.data_vis.geometry import bbox_corners, transform_points, world_to_ego


__all__ = ["generate_bev_grids", "read_grid_scene_meta"]

_INITIAL_MAP_BYTES = 64 * 1024 * 1024
_UNKNOWN_LANE_LAYER = 9
_HD_MAP_CACHE = {}


def generate_bev_grids(cfg: Config, scene_limit: int | None = None) -> dict:
    """生成全部（或前 scene_limit 个）场景的离线二值栅格，返回汇总统计。"""
    gen = cfg.data.bev_grid_generation
    scene_root = _repo_path(gen.scene_root)
    output_root = _repo_path(gen.output_dir)
    check_generation_paths(scene_root, output_root)
    scenes = sorted(path for path in scene_root.glob("scene_*") if path.is_dir())
    configured_limit = gen.max_scenes if scene_limit is None else scene_limit
    scenes = scenes[:configured_limit] if configured_limit and configured_limit > 0 else scenes
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = [(str(scene), str(output_root), cfg.model.world_model.grid, gen) for scene in scenes]
    results = _map_scenes(tasks, gen.num_workers)
    return {
        "scenes": len(results),
        "generated": sum(not item["skipped"] for item in results),
        "skipped": sum(item["skipped"] for item in results),
        "frames": sum(item["frames"] for item in results),
    }


def read_grid_scene_meta(scene_dir) -> dict:
    """只读一个栅格场景的 LMDB 元数据。"""
    env = lmdb.open(str(Path(scene_dir) / "lmdb"), readonly=True, subdir=True, lock=False)
    try:
        with env.begin() as txn:
            blob = txn.get(b"meta")
        if blob is None:
            raise RuntimeError("栅格场景缺少 meta：{}".format(scene_dir))
        return msgpack.unpackb(blob, raw=False)
    finally:
        env.close()


def _map_scenes(tasks, workers):
    if workers == 1:
        return [_generate_scene(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_generate_scene, tasks))


def _generate_scene(task):
    scene_text, output_text, grid_cfg, gen_cfg = task
    scene_dir = Path(scene_text)
    check_source_scene(scene_dir)
    final_dir = Path(output_text) / scene_dir.name
    if _is_complete(final_dir):
        meta = read_grid_scene_meta(final_dir)
        return {"scene": scene_dir.name, "frames": int(meta["num_frames"]), "skipped": True}

    temp_dir = final_dir.with_name(final_dir.name + ".tmp")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)
    try:
        frame_count = _write_scene(scene_dir, temp_dir, grid_cfg, gen_cfg)
        if final_dir.exists():
            raise FileExistsError("目标场景已存在但不完整，请先人工处理：{}".format(final_dir))
        os.replace(str(temp_dir), str(final_dir))
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise
    return {"scene": scene_dir.name, "frames": frame_count, "skipped": False}


def _write_scene(scene_dir, output_dir, grid_cfg, gen_cfg):
    source = _SourceScene(scene_dir)
    try:
        bev = _bev_params(grid_cfg)
        map_key = source.meta["map"].replace("_Opt", "")
        map_path = _repo_path(gen_cfg.map_dir) / gen_cfg.map_name_template.format(map=map_key)
        hd_map = _cached_hd_map(map_path)
        stop_lines = _stop_line_table(source.meta.get("traffic_lights", []))
        max_bytes = int(gen_cfg.lmdb_map_size_gb * 1024 ** 3)
        estimate = source.num_frames * ((len(grid_cfg.layer_names) * bev.height * bev.width + 7) // 8)
        map_size = min(max_bytes, max(_INITIAL_MAP_BYTES, int(estimate * 1.1)))
        env = lmdb.open(str(output_dir / "lmdb"), map_size=map_size, subdir=True)
        meta = _output_meta(source, grid_cfg, bev)
        with env.begin(write=True) as txn:
            txn.put(b"meta", msgpack.packb(meta, use_bin_type=True))
            txn.put(b"num_frames", msgpack.packb(source.num_frames))
        _write_frames(env, source, hd_map, stop_lines, bev, gen_cfg, len(grid_cfg.layer_names))
        env.sync()
        env.close()
        return source.num_frames
    finally:
        source.close()


def _write_frames(env, source, hd_map, stop_lines, bev, gen_cfg, layer_count):
    for index in range(source.num_frames):
        frame = source.frame(index)
        grid = _rasterize_frame(frame, hd_map, stop_lines, bev, gen_cfg, layer_count)
        packed = np.packbits(grid.reshape(-1), bitorder="little").tobytes()
        blob = zlib.compress(packed, gen_cfg.compress_level)
        with env.begin(write=True) as txn:
            txn.put("grid/{:08d}".format(index).encode(), blob)


def _rasterize_frame(frame, hd_map, stop_lines, bev, gen_cfg, layer_count):
    grid = np.zeros((layer_count, bev.height, bev.width), dtype=np.uint8)
    ego_pose = frame["ego"]["transform"]
    drivable, lane_classes = hd_map.drivable_lane_classes_bev(
        ego_pose, bev, gen_cfg.lane_half_width_m, gen_cfg.lane_line_width_m,
        gen_cfg.lane_type_to_layer, _UNKNOWN_LANE_LAYER)
    grid[5] = drivable
    grid[6:10] = np.equal(lane_classes[None], np.arange(6, 10)[:, None, None])
    _rasterize_agents(grid, frame.get("bboxes", []), ego_pose, bev, gen_cfg.agent_semantics)
    _rasterize_stop_lines(grid, stop_lines, frame.get("traffic_light_states", []), ego_pose,
                          bev, gen_cfg.stop_line_width_m)
    return grid


def _rasterize_agents(grid, boxes, ego_pose, bev, semantic_layers):
    w2e = world_to_ego(ego_pose)
    for semantic, layer in semantic_layers.items():
        selected = [box for box in boxes if box.get("semantic") == semantic]
        if not selected:
            continue
        corners = np.stack([bbox_corners(box)[[0, 1, 2, 3]] for box in selected])
        ego_corners = transform_points(corners.reshape(-1, 3), w2e).reshape(-1, 4, 3)
        rows, cols = ego_xy_to_pixel(ego_corners[..., :2], bev)
        polygons = np.stack((cols, rows), axis=-1).round().astype(np.int32)
        cv2.fillPoly(grid[layer], list(polygons), color=1)


def _stop_line_table(lights):
    rows = [(light["id"], stop["transform"][0], stop["transform"][1],
             stop["transform"][5], stop["lane_width"])
            for light in lights for stop in light.get("stop_waypoints", [])]
    return np.asarray(rows, dtype=np.float64).reshape(-1, 5)


def _rasterize_stop_lines(grid, table, states, ego_pose, bev, width_m):
    if table.size == 0:
        return
    state_by_id = {int(item["id"]): item.get("state") for item in states}
    state_layers = {"red": 0, "yellow": 1, "green": 2}
    yaw = np.radians(table[:, 3])
    lateral = np.stack((-np.sin(yaw), np.cos(yaw)), axis=1) * (table[:, 4:5] * 0.5)
    centers = table[:, 1:3]
    endpoints_xy = np.stack((centers - lateral, centers + lateral), axis=1)
    endpoints = np.concatenate((endpoints_xy, np.zeros((*endpoints_xy.shape[:2], 1))), axis=2)
    ego_points = transform_points(endpoints.reshape(-1, 3), world_to_ego(ego_pose)).reshape(-1, 2, 3)
    rows, cols = ego_xy_to_pixel(ego_points[..., :2], bev)
    segments = np.stack((cols, rows), axis=-1).round().astype(np.int32)
    px_per_m = bev.width / (bev.y_max - bev.y_min)
    thickness = max(int(round(width_m * px_per_m)), 1)
    for state, layer in state_layers.items():
        chosen = [segments[i].reshape(-1, 1, 2) for i, light_id in enumerate(table[:, 0])
                  if state_by_id.get(int(light_id)) == state]
        if chosen:
            cv2.polylines(grid[layer], chosen, False, color=1, thickness=thickness)


def _output_meta(source, grid_cfg, bev):
    return {
        "format": "bytedrive_bev_grid_v1",
        "complete": True,
        "source_scene": source.scene_dir.name,
        "source_map": source.meta["map"],
        "num_frames": source.num_frames,
        "shape": [len(grid_cfg.layer_names), bev.height, bev.width],
        "layer_names": list(grid_cfg.layer_names),
        "cell_size_m": float(grid_cfg.cell_size_m),
        "bounds_m": [bev.x_min, bev.x_max, bev.y_min, bev.y_max],
        "encoding": "packbits_little_zlib",
    }


def _bev_params(grid_cfg):
    height = int(round((grid_cfg.front_m + grid_cfg.rear_m) / grid_cfg.cell_size_m))
    width = int(round((grid_cfg.left_m + grid_cfg.right_m) / grid_cfg.cell_size_m))
    return BevParams(-grid_cfg.rear_m, grid_cfg.front_m, -grid_cfg.left_m,
                     grid_cfg.right_m, height, width)


def _is_complete(scene_dir):
    if not (scene_dir / "lmdb" / "data.mdb").is_file():
        return False
    try:
        return bool(read_grid_scene_meta(scene_dir).get("complete"))
    except Exception:
        return False


def _repo_path(value):
    path = Path(value)
    return path if path.is_absolute() else Path(__file__).resolve().parents[2] / path


def _cached_hd_map(path):
    """每个生成进程只解析同一 Town 的 HD Map 一次，跨场景复用折线与空间索引。"""
    key = str(Path(path).resolve())
    if key not in _HD_MAP_CACHE:
        _HD_MAP_CACHE[key] = HdMap(key)
    return _HD_MAP_CACHE[key]


class _SourceScene:
    def __init__(self, scene_dir):
        self.scene_dir = Path(scene_dir)
        self._env = lmdb.open(str(self.scene_dir / "lmdb"), readonly=True, subdir=True, lock=False)
        with self._env.begin() as txn:
            self.meta = msgpack.unpackb(txn.get(b"meta"), raw=False)
            self.num_frames = int(msgpack.unpackb(txn.get(b"num_frames")))

    def frame(self, index):
        with self._env.begin() as txn:
            return msgpack.unpackb(txn.get("{}/meta".format(index).encode()), raw=False)

    def close(self):
        self._env.close()
