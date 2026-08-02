"""多帧语义 LiDAR 静态融合、动态对象级重建、场景级断点恢复与批处理。

模块: data/multiframe_pointcloud_fusion/multiframe_pointcloud_fusion.py
依赖: hashlib, json, lmdb, msgpack, numpy, torch, time, config.schema,
      collector.writer.unpack_array, data.multiframe_pointcloud_fusion.checks
读取配置:
    multiframe_pointcloud_fusion.input_path / output_dir / device / frames_per_batch /
        placement_batch_size / voxel_size_m / box_fallback_margin_m /
        dynamic_frame_stride / moving_tags.pedestrian / moving_tags.vehicle
对外接口:
    - discover_scenes(input_path) -> list[Path]
    - fuse_scene(scene_dir, output_dir, cfg) -> Path
    - run_fusion(cfg, input_path=None, output_dir=None) -> dict
说明: 静态点仅按运动语义全局剔除；动态点优先按 obj_idx 对齐 actor，Box 仅作实例 ID
      失配回退，规避 CARLA Box 偏小问题。断点以场景为最小单位，场景内中断后整场重算。
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import lmdb
import msgpack
import numpy as np
import torch

from config.schema import Config
from data.multiframe_pointcloud_fusion.checks.multiframe_pointcloud_fusion_checks import (
    check_frame,
    check_run_checkpoint,
    check_input_path,
    check_output_dir,
    check_output_payload,
    check_scene_dir,
    check_scene_header,
)

# 采集器按独立 Py312 包运行；沿用可视化读取器的路径引导方式复用唯一数组解包实现。
_COLLECTOR_ROOT = Path(__file__).resolve().parents[1] / "carla_data_collector"
if str(_COLLECTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_COLLECTOR_ROOT))
from collector.writer import unpack_array  # noqa: E402


__all__ = ["discover_scenes", "fuse_scene", "run_fusion"]


_REPO_ROOT = Path(__file__).resolve().parents[2]
_STATIC_SOURCE = 0
_DYNAMIC_SOURCE = 1
_VEHICLE_CLASS = 0
_PEDESTRIAN_CLASS = 1
_OUTPUT_SCHEMA_VERSION = 2
_ATOMIC_REPLACE_ATTEMPTS = 5
_ATOMIC_REPLACE_DELAY_S = 0.05


class _SceneLmdb:
    """只读取融合所需的 LMDB 键，不打开 RGB 视频。"""

    def __init__(self, scene_dir):
        self.scene_dir = Path(scene_dir).resolve()
        check_scene_dir(self.scene_dir)
        self.lmdb_dir = self.scene_dir / "lmdb"
        self.env = lmdb.open(
            str(self.lmdb_dir), readonly=True, subdir=True, lock=False, readahead=False)
        with self.env.begin() as txn:
            self.meta_blob = txn.get(b"meta")
            frames_blob = txn.get(b"num_frames")
        if self.meta_blob is None or frames_blob is None:
            self.close()
            raise ValueError("场景 LMDB 缺少 meta/num_frames：{}".format(self.scene_dir))
        self.meta = msgpack.unpackb(self.meta_blob, raw=False)
        self.num_frames = int(msgpack.unpackb(frames_blob))
        check_scene_header(self.meta, self.num_frames)

    def frame(self, frame_index):
        """读取一帧轻量元数据与语义 LiDAR。"""
        with self.env.begin() as txn:
            meta_blob = txn.get(_key(frame_index, "meta"))
            lidar_blob = txn.get(_key(frame_index, "lidar"))
        if meta_blob is None or lidar_blob is None:
            raise ValueError("第 {} 帧缺少 meta/lidar".format(frame_index))
        frame_meta = msgpack.unpackb(meta_blob, raw=False)
        lidar = unpack_array(lidar_blob)
        check_frame(frame_index, frame_meta, lidar)
        return frame_meta, lidar

    def close(self):
        if getattr(self, "env", None) is not None:
            self.env.close()
            self.env = None

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        self.close()


def discover_scenes(input_path) -> list[Path]:
    """递归发现含 `lmdb/data.mdb` 的场景，并拒绝同名场景。

    参数:
        input_path: 单个场景目录或任意数据集根目录；相对路径以项目根解析
    返回:
        按绝对路径排序的场景目录列表
    """
    root = _resolve(input_path)
    check_input_path(root)
    scenes = (
        [root]
        if (root / "lmdb" / "data.mdb").is_file()
        else sorted({path.parent.parent.resolve() for path in root.rglob("lmdb/data.mdb")})
    )
    if not scenes:
        raise ValueError("输入目录下未发现 LMDB 场景：{}".format(root))
    names = [scene.name for scene in scenes]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError("递归输入存在重复场景名，拒绝覆盖：{}".format(duplicates))
    return scenes


def fuse_scene(scene_dir, output_dir, cfg: Config) -> Path:
    """融合一个场景并返回 `<场景名>.pt` 路径；已有同指纹结果直接复用。

    参数:
        scene_dir: 单场景目录，内部须有 lmdb/data.mdb
        output_dir: 最终 PT 与报告的项目内目录
        cfg: 项目完整 Config；读取 cfg.multiframe_pointcloud_fusion
    返回:
        完成或已存在的场景 PT 绝对路径
    """
    path, _status, _fingerprint_value = _fuse_scene(scene_dir, output_dir, cfg)
    return path


def run_fusion(cfg: Config, input_path=None, output_dir=None) -> dict:
    """处理单场景或递归数据集；按场景记录进度，失败后可从未完成场景继续。"""
    fusion = cfg.multiframe_pointcloud_fusion
    source = input_path if input_path is not None else fusion.input_path
    destination = _resolve(output_dir if output_dir is not None else fusion.output_dir)
    check_output_dir(destination, _REPO_ROOT)
    scenes = discover_scenes(source)
    destination.mkdir(parents=True, exist_ok=True)
    algorithm = _algorithm_config(fusion)
    checkpoint_path = destination / ".fusion_checkpoint.json"
    run_fingerprint = _run_fingerprint(source, scenes, algorithm)
    checkpoint = _prepare_run_checkpoint(
        checkpoint_path, run_fingerprint, [scene.name for scene in scenes])
    report = {
        "input": str(_resolve(source)),
        "output": str(destination),
        "completed": [],
        "skipped": [],
        "failed": [],
    }
    for scene in scenes:
        completed = checkpoint["completed"].get(scene.name)
        completed_path = Path(completed["output"]) if completed else None
        expected_path = destination / (scene.name + ".pt")
        if completed and completed_path.resolve() == expected_path.resolve() \
                and completed_path.is_file():
            output_stat = completed_path.stat()
            if output_stat.st_size == completed["output_size"] \
                    and output_stat.st_mtime_ns == completed["output_mtime_ns"]:
                report["skipped"].append({
                    "scene": scene.name, "output": str(completed_path)})
                continue
        checkpoint["completed"].pop(scene.name, None)
        checkpoint["current_scene"] = scene.name
        checkpoint["errors"].pop(scene.name, None)
        _atomic_json(checkpoint_path, checkpoint)
        try:
            path, status, scene_fingerprint = _fuse_scene(scene, destination, cfg)
            report[status].append({"scene": scene.name, "output": str(path)})
            output_stat = path.stat()
            checkpoint["completed"][scene.name] = {
                "output": str(path),
                "fingerprint": scene_fingerprint,
                "output_size": output_stat.st_size,
                "output_mtime_ns": output_stat.st_mtime_ns,
            }
            checkpoint["current_scene"] = None
        except Exception as exc:  # 批处理必须继续；完整错误落报告，CLI 最终返回非零
            error = "{}: {}".format(type(exc).__name__, exc)
            checkpoint["errors"][scene.name] = error
            checkpoint["current_scene"] = None
            report["failed"].append({
                "scene": scene.name,
                "path": str(scene),
                "error": error,
            })
        _atomic_json(checkpoint_path, checkpoint)
        _atomic_json(destination / "fusion_report.json", report)
    report["ok"] = not report["failed"]
    _atomic_json(destination / "fusion_report.json", report)
    if report["ok"]:
        _remove_run_checkpoint(checkpoint_path, destination)
    return report


def _fuse_scene(scene_dir, output_dir, cfg):
    scene_dir = Path(scene_dir).resolve()
    output_dir = Path(output_dir).resolve()
    check_scene_dir(scene_dir)
    check_output_dir(output_dir, _REPO_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    fusion = cfg.multiframe_pointcloud_fusion
    device = _device(fusion.device)
    with _SceneLmdb(scene_dir) as reader:
        signature = _scene_signature(reader)
        algorithm = _algorithm_config(fusion)
        fingerprint = _fingerprint(signature, algorithm)
        output_path = output_dir / (scene_dir.name + ".pt")
        if output_path.is_file():
            _check_existing_output(output_path, fingerprint)
            return output_path, "skipped", fingerprint
        chunks = _process_scene_batches(reader, fusion, device)
        payload = _finalize_scene(
            reader, chunks, fingerprint, signature, algorithm, fusion, device)
        _atomic_torch_save(output_path, payload)
        del payload
        written = torch.load(output_path, map_location="cpu", weights_only=False)
        check_output_payload(written, fingerprint)
        del written
    return output_path, "completed", fingerprint


def _process_scene_batches(reader, fusion, device):
    chunks = []
    start = 0
    while start < reader.num_frames:
        end = min(start + fusion.frames_per_batch, reader.num_frames)
        frames = [reader.frame(index) for index in range(start, end)]
        chunk = _process_batch(frames, start, reader.meta["lidar_extrinsic"], fusion, device)
        chunks.append(chunk)
        start = end
    return chunks


def _process_batch(frames, start_frame, lidar_extrinsic, fusion, device):
    point_counts = torch.tensor([len(lidar) for _, lidar in frames], device=device)
    frame_ids = torch.repeat_interleave(
        torch.arange(len(frames), device=device, dtype=torch.int64), point_counts)
    xyz = torch.cat([
        torch.from_numpy(_lidar_xyz(lidar)) for _, lidar in frames
    ]).to(device=device, dtype=torch.float32)
    tags = torch.cat([
        torch.from_numpy(lidar["obj_tag"].astype("int64", copy=False))
        for _, lidar in frames
    ]).to(device)
    object_ids = torch.cat([
        torch.from_numpy(lidar["obj_idx"].astype("int64", copy=False))
        for _, lidar in frames
    ]).to(device)
    ego_poses = torch.tensor(
        [meta["ego"]["transform"] for meta, _ in frames],
        dtype=torch.float32, device=device)
    ego_matrices = _pose_matrices(ego_poses)
    sensor_xyz = xyz + torch.tensor(lidar_extrinsic, dtype=torch.float32, device=device)
    world_xyz = _transform_indexed(sensor_xyz, ego_matrices, frame_ids)
    boxes = _boxes(frames, start_frame, device)
    moving = torch.isin(tags, _moving_tags(fusion, device))
    static_chunk = _aggregate(
        world_xyz[~moving], tags[~moving], None, fusion.voxel_size_m)
    dynamic_chunk, stats = _dynamic_observations(
        world_xyz, tags, object_ids, frame_ids, moving, boxes, fusion, device)
    return {
        "start_frame": start_frame,
        "end_frame": start_frame + len(frames),
        "ego_pose": ego_poses.detach().cpu(),
        "static": _cpu_tree(static_chunk),
        "dynamic": _cpu_tree(dynamic_chunk),
        "poses": _cpu_tree({
            "actor_id": boxes["actor_id"],
            "frame_index": boxes["frame_index"],
            "pose": boxes["pose"],
        }),
        "stats": {
            "input_points": int(len(world_xyz)),
            "static_points": int((~moving).sum().item()),
            "moving_removed": int(moving.sum().item()),
            **stats,
        },
    }


def _boxes(frames, start_frame, device):
    records = [
        (local_frame, box)
        for local_frame, (meta, _lidar) in enumerate(frames)
        for box in meta["bboxes"]
        if box.get("semantic") in ("vehicle", "pedestrian") and box.get("id") is not None
    ]
    ego_ids = [
        next((int(box["id"]) for box in meta["bboxes"]
              if box.get("semantic") == "ego" and box.get("id") is not None), -1)
        for meta, _lidar in frames
    ]
    if not records:
        return {
            "local_frame": torch.empty(0, dtype=torch.int64, device=device),
            "frame_index": torch.empty(0, dtype=torch.int64, device=device),
            "actor_id": torch.empty(0, dtype=torch.int64, device=device),
            "class_id": torch.empty(0, dtype=torch.int64, device=device),
            "pose": torch.empty((0, 6), dtype=torch.float32, device=device),
            "extent": torch.empty((0, 3), dtype=torch.float32, device=device),
            "ego_ids": torch.tensor(ego_ids, dtype=torch.int64, device=device),
        }
    local_frame = torch.tensor([item[0] for item in records], dtype=torch.int64, device=device)
    return {
        "local_frame": local_frame,
        "frame_index": local_frame + start_frame,
        "actor_id": torch.tensor(
            [int(item[1]["id"]) for item in records], dtype=torch.int64, device=device),
        "class_id": torch.tensor([
            _VEHICLE_CLASS if item[1]["semantic"] == "vehicle" else _PEDESTRIAN_CLASS
            for item in records
        ], dtype=torch.int64, device=device),
        "pose": torch.tensor([
            item[1]["location"] + item[1]["rotation"] for item in records
        ], dtype=torch.float32, device=device),
        "extent": torch.tensor([
            item[1]["extent"] for item in records
        ], dtype=torch.float32, device=device),
        "ego_ids": torch.tensor(ego_ids, dtype=torch.int64, device=device),
    }


def _dynamic_observations(world_xyz, tags, object_ids, point_frames, moving,
                          boxes, fusion, device):
    candidate = torch.nonzero(
        moving & (object_ids != boxes["ego_ids"][point_frames]), as_tuple=False).flatten()
    assigned_box = torch.full((len(candidate),), -1, dtype=torch.int64, device=device)
    primary = _primary_box_matches(
        tags[candidate], object_ids[candidate], point_frames[candidate], boxes, fusion)
    assigned_box[primary >= 0] = primary[primary >= 0]
    fallback_positions = torch.nonzero(assigned_box < 0, as_tuple=False).flatten()
    fallback_box = _fallback_box_matches(
        world_xyz[candidate[fallback_positions]], tags[candidate[fallback_positions]],
        point_frames[candidate[fallback_positions]], boxes, fusion)
    valid_fallback = fallback_box >= 0
    assigned_box[fallback_positions[valid_fallback]] = fallback_box[valid_fallback]
    assigned = assigned_box >= 0
    chosen_points = candidate[assigned]
    chosen_boxes = assigned_box[assigned]
    local_xyz = _world_to_box_local(
        world_xyz[chosen_points], boxes["pose"][chosen_boxes])
    dynamic = _aggregate(
        local_xyz, tags[chosen_points], boxes["actor_id"][chosen_boxes],
        fusion.voxel_size_m)
    primary_count = int((primary >= 0).sum().item())
    fallback_count = int(valid_fallback.sum().item())
    return dynamic, {
        "dynamic_id_matched": primary_count,
        "dynamic_box_fallback": fallback_count,
        "dynamic_unassigned": int(len(candidate) - primary_count - fallback_count),
        "ego_moving_removed": int((
            moving & (object_ids == boxes["ego_ids"][point_frames])).sum().item()),
    }


def _primary_box_matches(tags, object_ids, point_frames, boxes, fusion):
    result = torch.full_like(object_ids, -1)
    if not len(object_ids) or not len(boxes["actor_id"]):
        return result
    box_keys = _actor_keys(boxes["local_frame"], boxes["actor_id"])
    point_keys = _actor_keys(point_frames, object_ids)
    sorted_keys, order = torch.sort(box_keys)
    positions = torch.searchsorted(sorted_keys, point_keys)
    safe = positions.clamp_max(len(sorted_keys) - 1)
    valid = (positions < len(sorted_keys)) & (sorted_keys[safe] == point_keys)
    box_indices = order[safe]
    valid &= _tag_matches_class(tags, boxes["class_id"][box_indices], fusion)
    result[valid] = box_indices[valid]
    return result


def _fallback_box_matches(points, tags, point_frames, boxes, fusion):
    result = torch.full((len(points),), -1, dtype=torch.int64, device=points.device)
    if not len(points) or not len(boxes["actor_id"]):
        return result
    matrices = _pose_matrices(boxes["pose"])
    for frame in torch.unique(point_frames).tolist():
        point_indices = torch.nonzero(point_frames == frame, as_tuple=False).flatten()
        box_indices = torch.nonzero(boxes["local_frame"] == frame, as_tuple=False).flatten()
        if not len(point_indices) or not len(box_indices):
            continue
        delta = points[point_indices, None, :] - matrices[box_indices, :3, 3][None, :, :]
        local = torch.einsum("mbj,bjk->mbk", delta, matrices[box_indices, :3, :3])
        scale = boxes["extent"][box_indices] + fusion.box_fallback_margin_m
        normalized = local.abs() / scale[None, :, :]
        compatible = _tag_matches_class(
            tags[point_indices, None], boxes["class_id"][box_indices][None, :], fusion)
        inside = torch.all(normalized <= 1, dim=2) & compatible
        scores = normalized.square().sum(dim=2).masked_fill(~inside, torch.inf)
        best_score, best = scores.min(dim=1)
        valid = torch.isfinite(best_score)
        result[point_indices[valid]] = box_indices[best[valid]]
    return result


def _aggregate(points, tags, actor_ids, voxel_size):
    if voxel_size == 0:
        result = {"raw": True, "points": points, "tags": tags}
        if actor_ids is not None:
            result["actor_id"] = actor_ids
        return result
    voxels = torch.floor(points / voxel_size).to(torch.int64)
    columns = [tags[:, None]]
    if actor_ids is not None:
        columns.insert(0, actor_ids[:, None])
    keys = torch.cat(columns + [voxels], dim=1)
    unique, inverse = torch.unique(keys, dim=0, return_inverse=True)
    sums = torch.zeros((len(unique), 3), dtype=torch.float32, device=points.device)
    sums.scatter_add_(0, inverse[:, None].expand(-1, 3), points)
    counts = torch.bincount(inverse, minlength=len(unique)).to(torch.int64)
    return {"raw": False, "keys": unique, "sums": sums, "counts": counts}


def _finalize_scene(reader, chunks, fingerprint, signature, algorithm, fusion, device):
    ego_pose = torch.cat([chunk["ego_pose"] for chunk in chunks]).to(torch.float32)
    static_xyz, static_tags, _ = _merge_parts(
        [chunk["static"] for chunk in chunks], fusion.voxel_size_m, device, dynamic=False)
    local_xyz, dynamic_tags, model_actors = _merge_parts(
        [chunk["dynamic"] for chunk in chunks], fusion.voxel_size_m, device, dynamic=True)
    poses = _concat_pose_parts([chunk["poses"] for chunk in chunks])
    dynamic_xyz, placed_tags, placed_actors, frame_indices = _place_dynamic(
        local_xyz, dynamic_tags, model_actors, poses, fusion, device)
    static_count = len(static_xyz)
    dynamic_count = len(dynamic_xyz)
    xyz = torch.cat((static_xyz, dynamic_xyz)).to(torch.float32)
    obj_tag = torch.cat((static_tags, placed_tags)).to(torch.uint8)
    source = torch.cat((
        torch.full((static_count,), _STATIC_SOURCE, dtype=torch.uint8),
        torch.full((dynamic_count,), _DYNAMIC_SOURCE, dtype=torch.uint8),
    ))
    actor_id = torch.cat((
        torch.full((static_count,), -1, dtype=torch.int64), placed_actors.to(torch.int64),
    ))
    frame_index = torch.cat((
        torch.full((static_count,), -1, dtype=torch.int32), frame_indices.to(torch.int32),
    ))
    stats = _sum_stats([chunk["stats"] for chunk in chunks])
    stats.update({
        "output_static_points": static_count,
        "canonical_dynamic_points": len(local_xyz),
        "output_dynamic_points": dynamic_count,
        "output_total_points": len(xyz),
    })
    return {
        "xyz": xyz.contiguous(),
        "obj_tag": obj_tag.contiguous(),
        "source": source.contiguous(),
        "actor_id": actor_id.contiguous(),
        "frame_index": frame_index.contiguous(),
        "ego_pose": ego_pose.contiguous(),
        "metadata": {
            "scene_name": reader.scene_dir.name,
            "coordinate_frame": "carla_world",
            "ego_pose_fields": ["x", "y", "z", "roll", "pitch", "yaw"],
            "ego_pose_indexing": "row_index_equals_frame_index",
            "ego_box_reconstructed": False,
            "source_codes": {"static": _STATIC_SOURCE, "dynamic": _DYNAMIC_SOURCE},
            "fingerprint": fingerprint,
            "input_signature": signature,
            "fusion_config": algorithm,
            "scene_meta": reader.meta,
            "stats": stats,
        },
    }


def _merge_parts(parts, voxel_size, device, dynamic):
    if voxel_size == 0:
        points = _cat_or_empty([part["points"] for part in parts], (0, 3), torch.float32)
        tags = _cat_or_empty([part["tags"] for part in parts], (0,), torch.int64)
        actors = (
            _cat_or_empty([part["actor_id"] for part in parts], (0,), torch.int64)
            if dynamic else None)
        return points, tags, actors
    keys = _cat_or_empty([part["keys"] for part in parts],
                         (0, 5 if dynamic else 4), torch.int64).to(device)
    sums = _cat_or_empty([part["sums"] for part in parts], (0, 3), torch.float32).to(device)
    counts = _cat_or_empty([part["counts"] for part in parts], (0,), torch.int64).to(device)
    if not len(keys):
        return (torch.empty((0, 3), dtype=torch.float32),
                torch.empty(0, dtype=torch.int64),
                torch.empty(0, dtype=torch.int64) if dynamic else None)
    unique, inverse = torch.unique(keys, dim=0, return_inverse=True)
    merged_sums = torch.zeros((len(unique), 3), dtype=torch.float32, device=device)
    merged_sums.scatter_add_(0, inverse[:, None].expand(-1, 3), sums)
    merged_counts = torch.zeros(len(unique), dtype=torch.int64, device=device)
    merged_counts.scatter_add_(0, inverse, counts)
    points = (merged_sums / merged_counts[:, None]).cpu()
    unique = unique.cpu()
    if dynamic:
        return points, unique[:, 1], unique[:, 0]
    return points, unique[:, 0], None


def _place_dynamic(local_xyz, tags, actors, poses, fusion, device):
    empty_xyz = torch.empty((0, 3), dtype=torch.float32)
    empty_i64 = torch.empty(0, dtype=torch.int64)
    if not len(local_xyz) or not len(poses["actor_id"]):
        return empty_xyz, empty_i64, empty_i64, empty_i64
    order = torch.argsort(actors)
    local_xyz = local_xyz[order].to(device)
    tags = tags[order].to(device)
    actors = actors[order].to(device)
    unique_actors, counts = torch.unique_consecutive(actors, return_counts=True)
    offsets = torch.cumsum(counts, dim=0) - counts
    pose_keep = poses["frame_index"] % fusion.dynamic_frame_stride == 0
    pose_actor = poses["actor_id"][pose_keep]
    pose_frame = poses["frame_index"][pose_keep]
    pose_values = poses["pose"][pose_keep]
    xyz_parts, tag_parts, actor_parts, frame_parts = [], [], [], []
    for start in range(0, len(pose_actor), fusion.placement_batch_size):
        end = min(start + fusion.placement_batch_size, len(pose_actor))
        batch_actor = pose_actor[start:end].to(device)
        positions = torch.searchsorted(unique_actors, batch_actor)
        safe = positions.clamp_max(len(unique_actors) - 1)
        valid = (positions < len(unique_actors)) & (unique_actors[safe] == batch_actor)
        if not bool(valid.any()):
            continue
        batch_actor = batch_actor[valid]
        batch_frame = pose_frame[start:end].to(device)[valid]
        batch_pose = pose_values[start:end].to(device)[valid]
        model_positions = safe[valid]
        batch_counts = counts[model_positions]
        pose_indices = torch.repeat_interleave(
            torch.arange(len(batch_actor), device=device), batch_counts)
        output_offsets = torch.cumsum(batch_counts, dim=0) - batch_counts
        local_offsets = torch.arange(int(batch_counts.sum().item()), device=device) \
            - torch.repeat_interleave(output_offsets, batch_counts)
        model_indices = torch.repeat_interleave(offsets[model_positions], batch_counts) \
            + local_offsets
        matrices = _pose_matrices(batch_pose)
        placed = _transform_indexed(local_xyz[model_indices], matrices, pose_indices)
        xyz_parts.append(placed.cpu())
        tag_parts.append(tags[model_indices].cpu())
        actor_parts.append(batch_actor[pose_indices].cpu())
        frame_parts.append(batch_frame[pose_indices].cpu())
    return (
        _cat_or_empty(xyz_parts, (0, 3), torch.float32),
        _cat_or_empty(tag_parts, (0,), torch.int64),
        _cat_or_empty(actor_parts, (0,), torch.int64),
        _cat_or_empty(frame_parts, (0,), torch.int64),
    )


def _concat_pose_parts(parts):
    return {
        "actor_id": _cat_or_empty([part["actor_id"] for part in parts], (0,), torch.int64),
        "frame_index": _cat_or_empty(
            [part["frame_index"] for part in parts], (0,), torch.int64),
        "pose": _cat_or_empty([part["pose"] for part in parts], (0, 6), torch.float32),
    }


def _pose_matrices(poses):
    poses = poses.to(torch.float32)
    radians = torch.deg2rad(poses[:, 3:6])
    cr, cp, cy = torch.cos(radians).unbind(dim=1)
    sr, sp, sy = torch.sin(radians).unbind(dim=1)
    matrices = torch.zeros((len(poses), 4, 4), dtype=torch.float32, device=poses.device)
    matrices[:, 3, 3] = 1
    matrices[:, :3, 3] = poses[:, :3]
    matrices[:, 0, 0] = cp * cy
    matrices[:, 0, 1] = cy * sp * sr - sy * cr
    matrices[:, 0, 2] = -cy * sp * cr - sy * sr
    matrices[:, 1, 0] = sy * cp
    matrices[:, 1, 1] = sy * sp * sr + cy * cr
    matrices[:, 1, 2] = -sy * sp * cr + cy * sr
    matrices[:, 2, 0] = sp
    matrices[:, 2, 1] = -cp * sr
    matrices[:, 2, 2] = cp * cr
    return matrices


def _transform_indexed(points, matrices, indices):
    rotations = matrices[indices, :3, :3]
    translations = matrices[indices, :3, 3]
    return torch.bmm(rotations, points.unsqueeze(2)).squeeze(2) + translations


def _world_to_box_local(points, box_poses):
    matrices = _pose_matrices(box_poses)
    delta = points - matrices[:, :3, 3]
    return torch.bmm(delta.unsqueeze(1), matrices[:, :3, :3]).squeeze(1)


def _tag_matches_class(tags, class_ids, fusion):
    vehicle = torch.isin(tags, torch.tensor(
        fusion.moving_tags.vehicle, dtype=torch.int64, device=tags.device))
    pedestrian = torch.isin(tags, torch.tensor(
        fusion.moving_tags.pedestrian, dtype=torch.int64, device=tags.device))
    return torch.where(class_ids == _VEHICLE_CLASS, vehicle, pedestrian)


def _moving_tags(fusion, device):
    return torch.tensor(
        fusion.moving_tags.pedestrian + fusion.moving_tags.vehicle,
        dtype=torch.int64, device=device)


def _actor_keys(frame_indices, actor_ids):
    return (frame_indices.to(torch.int64) << 32) | (actor_ids.to(torch.int64) & 0xFFFFFFFF)


def _lidar_xyz(lidar):
    return np.stack((lidar["x"], lidar["y"], lidar["z"]), axis=1)


def _scene_signature(reader):
    data_path = reader.lmdb_dir / "data.mdb"
    stat = data_path.stat()
    return {
        "scene_name": reader.scene_dir.name,
        "num_frames": reader.num_frames,
        "data_mdb_size": stat.st_size,
        "data_mdb_mtime_ns": stat.st_mtime_ns,
        "meta_sha256": hashlib.sha256(reader.meta_blob).hexdigest(),
    }


def _algorithm_config(fusion):
    values = asdict(fusion)
    values.pop("input_path")
    values.pop("output_dir")
    values["output_schema_version"] = _OUTPUT_SCHEMA_VERSION
    return values


def _fingerprint(signature, algorithm):
    raw = json.dumps(
        {"input": signature, "algorithm": algorithm},
        ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _run_fingerprint(source, scenes, algorithm):
    def descriptor(scene):
        stat = (scene / "lmdb" / "data.mdb").stat()
        return {
            "name": scene.name,
            "path": str(scene),
            "data_mdb_size": stat.st_size,
            "data_mdb_mtime_ns": stat.st_mtime_ns,
        }
    descriptors = [descriptor(scene) for scene in scenes]
    raw = json.dumps({
        "input": str(_resolve(source)),
        "scenes": descriptors,
        "algorithm": algorithm,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _prepare_run_checkpoint(path, fingerprint, scene_names):
    if path.is_file():
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        check_run_checkpoint(checkpoint, fingerprint, scene_names)
        return checkpoint
    checkpoint = {
        "fingerprint": fingerprint,
        "scene_names": scene_names,
        "completed": {},
        "current_scene": None,
        "errors": {},
    }
    _atomic_json(path, checkpoint)
    return checkpoint


def _check_existing_output(output_path, fingerprint):
    payload = torch.load(output_path, map_location="cpu", weights_only=False)
    check_output_payload(payload, fingerprint)


def _remove_run_checkpoint(path, output_dir):
    checkpoint_path = Path(path).resolve()
    expected = (Path(output_dir) / ".fusion_checkpoint.json").resolve()
    if checkpoint_path != expected:
        raise RuntimeError("拒绝清理非预期的场景级断点文件")
    if checkpoint_path.is_file():
        checkpoint_path.unlink()


def _atomic_torch_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-{}".format(os.getpid()))
    try:
        torch.save(payload, temporary)
        _replace_file(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-{}".format(os.getpid()))
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _replace_file(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sum_stats(stats_parts):
    keys = {key for part in stats_parts for key in part}
    return {key: sum(int(part.get(key, 0)) for part in stats_parts) for key in sorted(keys)}


def _replace_file(source, destination):
    for attempt in range(_ATOMIC_REPLACE_ATTEMPTS):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt + 1 == _ATOMIC_REPLACE_ATTEMPTS:
                raise
            time.sleep(_ATOMIC_REPLACE_DELAY_S * (attempt + 1))


def _cpu_tree(tree):
    return {key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in tree.items()}


def _cat_or_empty(parts, shape, dtype):
    nonempty = [part for part in parts if len(part)]
    return torch.cat(nonempty) if nonempty else torch.empty(shape, dtype=dtype)


def _key(*parts):
    return "/".join(str(part) for part in parts).encode("utf-8")


def _resolve(path):
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (_REPO_ROOT / candidate).resolve()


def _device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求 CUDA，但当前 PyTorch/CUDA 不可用")
    return device
