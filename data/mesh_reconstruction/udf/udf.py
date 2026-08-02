"""以 PyTorch 稀疏规则张量构建静态世界与动态局部截断无符号距离场。

模块: data/mesh_reconstruction/udf/udf.py
依赖: dataclasses, hashlib, itertools, json, os, pathlib, numpy, open3d, torch
读取配置: mesh_reconstruction.udf、device/tensor_batch_size/tag_candidate_budget
对外接口:
    - build_sparse_udf(points, tags, cfg, device, batch_size, candidate_budget,
                       max_voxels) -> dict
    - reconstruct_udf_scene(input_path, output_path, cfg) -> Path
    - run_udf_reconstruction(cfg, input_path=None, output_dir=None, force=False) -> dict
说明: 体素坐标表示网格单元，中心为 (coord+0.5)*voxel_size；UDF 使用最近局部
      切平面绝对距离，并由最近点欧氏距离限制在观测窄带内。
"""

from dataclasses import asdict
import hashlib
from itertools import product
import json
import os
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

from data.mesh_reconstruction.checks.mesh_reconstruction_checks import (
    check_input_path,
    check_output_dir,
    check_source_payload,
)
from data.mesh_reconstruction.mesh_reconstruction import discover_pointclouds
from data.mesh_reconstruction.udf.checks.udf_checks import (
    check_packed_dynamic,
    check_udf_field,
    check_udf_inputs,
    check_udf_output_path,
    check_udf_payload,
)

__all__ = ["build_sparse_udf", "reconstruct_udf_scene", "run_udf_reconstruction"]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCHEMA_VERSION = 1


def build_sparse_udf(points, tags, cfg, device, batch_size, candidate_budget,
                     max_voxels):
    """把点云编码为观测窄带稀疏 TUDF，返回 CPU 连续张量。"""
    check_udf_inputs(points, tags)
    if not len(points):
        return _empty_field(cfg)
    points, tags, cells, counts = _voxel_aggregate(
        points.to(device), tags.to(device), cfg.voxel_size_m)
    normals = _estimate_normals(points, cfg).to(device)
    queries = _expand_cells(
        cells, cfg.band_width_voxels, candidate_budget, max_voxels)
    centers = (queries.to(torch.float32) + 0.5) * cfg.voxel_size_m
    nearest, distance = _nearest_surface(
        queries, centers, cells, points, cfg.band_width_voxels,
        batch_size, candidate_budget)
    keep = distance <= cfg.truncation_m
    if not bool(keep.any()):
        raise RuntimeError("TUDF 截断带未保留任何体素")
    queries, centers, nearest, distance = (
        queries[keep], centers[keep], nearest[keep], distance[keep])
    delta = centers - points[nearest]
    selected_normals = normals[nearest]
    normal_valid = torch.linalg.vector_norm(selected_normals, dim=1) > 0.5
    plane_distance = torch.abs((delta * selected_normals).sum(dim=1))
    udf = torch.where(normal_valid, plane_distance, distance).clamp_max(cfg.truncation_m)
    weight = (1 - distance / cfg.truncation_m).clamp(0, 1)
    result = {
        "voxel_coords": queries.to(torch.int32).cpu().contiguous(),
        "udf": udf.to(torch.float32).cpu().contiguous(),
        "weight": weight.to(torch.float32).cpu().contiguous(),
        "observation_count": counts[nearest].to(torch.int32).cpu().contiguous(),
        "obj_tag": tags[nearest].to(torch.uint8).cpu().contiguous(),
        "normal": selected_normals.to(torch.float32).cpu().contiguous(),
        "voxel_size_m": torch.tensor(cfg.voxel_size_m, dtype=torch.float32),
        "truncation_m": torch.tensor(cfg.truncation_m, dtype=torch.float32),
    }
    check_udf_field(result)
    return result


def reconstruct_udf_scene(input_path, output_path, cfg):
    """构建一个场景的静态/动态 TUDF 并原子保存统一 PT。"""
    source, destination = _resolve(input_path), _resolve(output_path)
    check_input_path(source)
    check_udf_output_path(destination, _REPO_ROOT)
    payload = torch.load(source, map_location="cpu", weights_only=True)
    check_source_payload(payload)
    reconstruction = cfg.mesh_reconstruction
    device = _device(reconstruction.device)
    static = build_sparse_udf(
        payload["static"]["xyz"], payload["static"]["obj_tag"],
        reconstruction.udf.static, device, reconstruction.tensor_batch_size,
        reconstruction.tag_candidate_budget, reconstruction.udf.max_voxels)
    dynamic = _build_dynamic(
        payload["dynamic_objects"], reconstruction.udf.dynamic, device,
        reconstruction.tensor_batch_size, reconstruction.tag_candidate_budget,
        reconstruction.udf.max_voxels)
    source_fingerprint = _source_fingerprint(source, payload["metadata"])
    algorithm = asdict(reconstruction.udf)
    fingerprint = _hash_json({"source": source_fingerprint, "algorithm": algorithm})
    result = {
        "static_udf": static,
        "dynamic_udfs": dynamic,
        "dynamic_poses": {key: value.clone().contiguous()
                          for key, value in payload["dynamic_poses"].items()},
        "ego_pose": payload["ego_pose"].clone().contiguous(),
        "metadata": {
            "schema_version": _SCHEMA_VERSION,
            "representation": "sparse_tudf",
            "scene_name": str(payload["metadata"].get("scene_name", source.stem)),
            "coordinate_frames": {
                "static_udf.voxel_coords": "carla_world_grid",
                "dynamic_udfs.voxel_coords_local": "actor_box_local_grid",
                "dynamic_poses.transform": "carla_world",
                "ego_pose": "carla_world",
            },
            "voxel_center_formula": "(voxel_coord + 0.5) * voxel_size_m",
            "distance_definition": "nearest_local_tangent_plane_unsigned_distance",
            "source_path": str(source),
            "source_fingerprint": source_fingerprint,
            "reconstruction_fingerprint": fingerprint,
            "reconstruction_config": algorithm,
            "source_metadata": payload["metadata"],
            "stats": {
                "static_voxels": len(static["voxel_coords"]),
                "dynamic_voxels": len(dynamic["voxel_coords_local"]),
                "dynamic_objects": len(dynamic["actor_id"]),
                "observed_dynamic_objects": int(
                    (dynamic["voxel_offsets"][1:] > dynamic["voxel_offsets"][:-1]).sum()),
            },
        },
    }
    check_udf_payload(result)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(destination, result)
    check_udf_payload(torch.load(destination, map_location="cpu", weights_only=True))
    return destination


def run_udf_reconstruction(cfg, input_path=None, output_dir=None, force=False):
    """递归处理融合 PT，使用独立检查点和报告支持跳过与失败继续。"""
    reconstruction = cfg.mesh_reconstruction
    source = _resolve(input_path if input_path is not None else reconstruction.input_path)
    destination = _resolve(output_dir if output_dir is not None else reconstruction.output_dir)
    check_input_path(source)
    check_output_dir(destination, source, _REPO_ROOT)
    files = discover_pointclouds(source)
    destination.mkdir(parents=True, exist_ok=True)
    root = source if source.is_dir() else source.parent
    algorithm = asdict(reconstruction.udf)
    report = {"representation": "sparse_tudf", "input": str(source),
              "output": str(destination), "completed": [], "skipped": [], "failed": []}
    checkpoint_path = destination / ".udf_checkpoint.json"
    checkpoint = {"algorithm_fingerprint": _hash_json(algorithm), "completed": {}, "errors": {}}
    for path in files:
        relative = path.relative_to(root)
        output = destination / relative.parent / (path.stem + ".udf.pt")
        key = relative.as_posix()
        try:
            if not force and _existing_matches(path, output, algorithm):
                report["skipped"].append({"input": str(path), "output": str(output)})
            else:
                reconstruct_udf_scene(path, output, cfg)
                report["completed"].append({"input": str(path), "output": str(output)})
            checkpoint["completed"][key] = str(output)
            checkpoint["errors"].pop(key, None)
        except Exception as exc:
            error = "{}: {}".format(type(exc).__name__, exc)
            checkpoint["errors"][key] = error
            report["failed"].append({"input": str(path), "output": str(output), "error": error})
        _atomic_json(checkpoint_path, checkpoint)
        _atomic_json(destination / "udf_report.json", report)
    report["ok"] = not report["failed"]
    _atomic_json(destination / "udf_report.json", report)
    return report


def _voxel_aggregate(points, tags, voxel_size):
    cells = torch.floor(points / voxel_size).to(torch.int64)
    unique, inverse = torch.unique(cells, dim=0, return_inverse=True)
    counts = torch.bincount(inverse, minlength=len(unique))
    sums = torch.zeros((len(unique), 3), dtype=torch.float32, device=points.device)
    sums.scatter_add_(0, inverse[:, None].expand(-1, 3), points)
    centroids = sums / counts[:, None]
    pairs, pair_counts = torch.unique(
        inverse * 256 + tags.to(torch.int64), return_counts=True)
    pair_voxel, pair_tag = pairs // 256, pairs % 256
    maximum = torch.zeros(len(unique), dtype=pair_counts.dtype, device=points.device)
    maximum.scatter_reduce_(0, pair_voxel, pair_counts, reduce="amax", include_self=True)
    best = pair_counts == maximum[pair_voxel]
    voxel_tags = torch.full((len(unique),), 255, dtype=torch.int64, device=points.device)
    voxel_tags.scatter_reduce_(
        0, pair_voxel[best], pair_tag[best], reduce="amin", include_self=True)
    return centroids, voxel_tags.to(torch.uint8), unique, counts


def _estimate_normals(points, cfg):
    if len(points) < 3:
        return torch.zeros_like(points, device="cpu")
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
        points.cpu().numpy().astype(np.float64)))
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=cfg.normal_radius_m, max_nn=cfg.normal_max_nn))
    normals = torch.from_numpy(np.asarray(cloud.normals).copy()).to(torch.float32)
    return torch.nn.functional.normalize(normals, dim=1)


def _expand_cells(cells, band, candidate_budget, max_voxels):
    if band == 0:
        return cells
    lower, upper = cells.min(dim=0).values - band, cells.max(dim=0).values + band
    shape = upper - lower + 1
    offsets = torch.tensor(list(product(range(-band, band + 1), repeat=3)),
                           dtype=torch.int64, device=cells.device)
    chunk = max(1, candidate_budget // len(offsets))
    parts = []
    for start in range(0, len(cells), chunk):
        expanded = cells[start:start + chunk, None, :] + offsets[None, :, :]
        parts.append(torch.unique(_linear_keys(expanded.reshape(-1, 3), lower, shape)))
    keys = torch.unique(torch.cat(parts))
    if len(keys) > max_voxels:
        raise RuntimeError("稀疏 TUDF 体素数 {:,} 超过安全上限 {:,}".format(
            len(keys), max_voxels))
    return _decode_keys(keys, lower, shape)


def _nearest_surface(query_cells, centers, surface_cells, points, band,
                     batch_size, candidate_budget):
    lower = torch.minimum(query_cells.min(dim=0).values, surface_cells.min(dim=0).values)
    upper = torch.maximum(query_cells.max(dim=0).values, surface_cells.max(dim=0).values)
    shape = upper - lower + 1
    keys = _linear_keys(surface_cells, lower, shape)
    keys, order = torch.sort(keys)
    sorted_points = points[order]
    reach = max(1, band)
    offsets = torch.tensor(list(product(range(-reach, reach + 1), repeat=3)),
                           dtype=torch.int64, device=query_cells.device)
    chunk = max(1, min(batch_size, candidate_budget // len(offsets)))
    nearest_parts, distance_parts = [], []
    for start in range(0, len(query_cells), chunk):
        cells = query_cells[start:start + chunk]
        candidates = cells[:, None, :] + offsets[None, :, :]
        valid = ((candidates >= lower) & (candidates <= upper)).all(dim=2)
        query_keys = _linear_keys(candidates, lower, shape)
        positions = torch.searchsorted(keys, query_keys)
        safe = positions.clamp(max=len(keys) - 1)
        matches = valid & (positions < len(keys)) & (keys[safe] == query_keys)
        distances = torch.linalg.vector_norm(
            sorted_points[safe] - centers[start:start + chunk, None, :], dim=2)
        distances = torch.where(matches, distances, torch.inf)
        best_distance, best = distances.min(dim=1)
        if bool(torch.isinf(best_distance).any()):
            raise RuntimeError("TUDF 邻域查询未找到生成该体素的表面点")
        nearest_parts.append(order[safe.gather(1, best[:, None]).squeeze(1)])
        distance_parts.append(best_distance)
    return torch.cat(nearest_parts), torch.cat(distance_parts)


def _build_dynamic(objects, cfg, device, batch_size, candidate_budget, max_voxels):
    fields = []
    offsets = objects["point_offsets"]
    for index in range(len(objects["actor_id"])):
        first, last = int(offsets[index]), int(offsets[index + 1])
        fields.append(build_sparse_udf(
            objects["xyz_local"][first:last], objects["obj_tag"][first:last], cfg,
            device, batch_size, candidate_budget, max_voxels))
    counts = torch.tensor([len(field["voxel_coords"]) for field in fields], dtype=torch.int64)
    voxel_offsets = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0)))
    packed = {
        "actor_id": objects["actor_id"].clone().contiguous(),
        "class_id": objects["class_id"].clone().contiguous(),
        "extent": objects["extent"].clone().contiguous(),
        "voxel_offsets": voxel_offsets,
        "voxel_coords_local": _cat(fields, "voxel_coords", (0, 3), torch.int32),
        "udf": _cat(fields, "udf", (0,), torch.float32),
        "weight": _cat(fields, "weight", (0,), torch.float32),
        "observation_count": _cat(fields, "observation_count", (0,), torch.int32),
        "obj_tag": _cat(fields, "obj_tag", (0,), torch.uint8),
        "normal_local": _cat(fields, "normal", (0, 3), torch.float32),
        "voxel_size_m": torch.tensor(cfg.voxel_size_m, dtype=torch.float32),
        "truncation_m": torch.tensor(cfg.truncation_m, dtype=torch.float32),
    }
    check_packed_dynamic(packed)
    return packed


def _cat(fields, key, shape, dtype):
    values = [field[key] for field in fields if len(field["voxel_coords"])]
    return torch.cat(values).contiguous() if values else torch.empty(shape, dtype=dtype)


def _empty_field(cfg):
    return {
        "voxel_coords": torch.empty((0, 3), dtype=torch.int32),
        "udf": torch.empty(0, dtype=torch.float32),
        "weight": torch.empty(0, dtype=torch.float32),
        "observation_count": torch.empty(0, dtype=torch.int32),
        "obj_tag": torch.empty(0, dtype=torch.uint8),
        "normal": torch.empty((0, 3), dtype=torch.float32),
        "voxel_size_m": torch.tensor(cfg.voxel_size_m, dtype=torch.float32),
        "truncation_m": torch.tensor(cfg.truncation_m, dtype=torch.float32),
    }


def _linear_keys(cells, lower, shape):
    shifted = cells - lower
    return (shifted[..., 0] * shape[1] + shifted[..., 1]) * shape[2] + shifted[..., 2]


def _decode_keys(keys, lower, shape):
    yz = shape[1] * shape[2]
    x, remainder = keys // yz, keys % yz
    y, z = remainder // shape[2], remainder % shape[2]
    return torch.stack((x, y, z), dim=1) + lower


def _device(spec):
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求 CUDA，但当前 PyTorch 不支持 CUDA")
    return device


def _resolve(path):
    value = Path(path)
    return value.resolve() if value.is_absolute() else (_REPO_ROOT / value).resolve()


def _source_fingerprint(path, metadata):
    stat = path.stat()
    return _hash_json({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                       "fusion_fingerprint": metadata.get("fingerprint", "")})


def _existing_matches(source, output, algorithm):
    if not output.is_file():
        return False
    try:
        source_payload = torch.load(source, map_location="cpu", weights_only=True)
        result = torch.load(output, map_location="cpu", weights_only=True)
        expected = _hash_json({
            "source": _source_fingerprint(source, source_payload["metadata"]),
            "algorithm": algorithm,
        })
        return result.get("metadata", {}).get("reconstruction_fingerprint") == expected
    except Exception:
        return False


def _hash_json(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_torch_save(path, payload):
    temporary = path.with_name("{}.tmp.{}".format(path.name, os.getpid()))
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("{}.tmp.{}".format(path.name, os.getpid()))
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
