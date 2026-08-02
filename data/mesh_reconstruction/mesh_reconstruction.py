"""融合 PT 的静态优先 Mesh 重建、动态 donor 复用、断点恢复与批处理。

模块: data/mesh_reconstruction/mesh_reconstruction.py
依赖: dataclasses, hashlib, json, os, pathlib, time, torch, config.schema,
      data.mesh_reconstruction.surface/dynamic/checks
读取配置: mesh_reconstruction 全树
对外接口:
    - discover_pointclouds(input_path) -> list[Path]
    - reconstruct_scene(input_path, output_path, cfg) -> Path
    - run_reconstruction(cfg, input_path=None, output_dir=None, force=False) -> dict
说明: 每个场景先完成静态 Mesh，再进入动态重建；可选水密修复默认关闭，无观测支撑
      的 Poisson 外推表面默认裁除。统一 PT 保留规范局部模型与原始逐帧位姿。
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch

from config.schema import Config
from data.mesh_reconstruction.checks.mesh_reconstruction_checks import (
    check_input_path,
    check_output_dir,
    check_output_path,
    check_output_payload,
    check_source_payload,
)
from data.mesh_reconstruction.dynamic import reconstruct_dynamic_objects
from data.mesh_reconstruction.surface import reconstruct_surface_isolated

__all__ = ["discover_pointclouds", "reconstruct_scene", "run_reconstruction"]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA_VERSION = 1
_METHOD_NAMES = ["unobserved", "poisson", "exact_reuse", "similar_reuse", "box"]
_REPLACE_ATTEMPTS = 5
_REPLACE_DELAY_S = 0.05


def discover_pointclouds(input_path) -> list[Path]:
    """递归发现融合 PT，并排除本模块产出的 `.mesh.pt`。"""
    source = _resolve(input_path)
    check_input_path(source)
    files = [source] if source.is_file() else sorted(
        path.resolve() for path in source.rglob("*.pt")
        if not path.name.endswith(".mesh.pt"))
    if not files:
        raise ValueError("输入目录下未发现融合 .pt: {}".format(source))
    return files


def reconstruct_scene(input_path, output_path, cfg: Config) -> Path:
    """按静态、动态顺序重建一个融合 PT，并原子保存统一 Mesh PT。"""
    source, destination = _resolve(input_path), _resolve(output_path)
    check_input_path(source)
    assert source.is_file(), "reconstruct_scene 仅接受单个融合 PT"
    check_output_path(destination, _REPO_ROOT)
    payload = torch.load(source, map_location="cpu", weights_only=True)
    check_source_payload(payload)
    reconstruction = cfg.mesh_reconstruction
    device = _device(reconstruction.device)
    source_fingerprint = _source_fingerprint(source, payload["metadata"])
    algorithm = _algorithm_config(reconstruction)
    fingerprint = _hash_json({"source": source_fingerprint, "algorithm": algorithm})
    work_dir = destination.parent / ".mesh_work"
    try:
        lidar_origins = _lidar_origins(payload)
        static = reconstruct_surface_isolated(
            payload["static"]["xyz"], payload["static"]["obj_tag"], lidar_origins,
            reconstruction.static, reconstruction.repair.static_hole_radius_m,
            reconstruction.repair.enabled,
            device, reconstruction.tensor_batch_size, reconstruction.tag_candidate_budget,
            reconstruction.poisson_threads, work_dir)
        fusion_voxel = float(payload["metadata"]["fusion_config"]["voxel_size_m"])
        source_voxel = fusion_voxel if fusion_voxel > 0 else reconstruction.dynamic.voxel_size_m
        dynamic, dynamic_diagnostics = reconstruct_dynamic_objects(
            payload["dynamic_objects"], source_voxel, reconstruction, device, work_dir)
    finally:
        _remove_empty_dir(work_dir)
    result = {
        "static_mesh": static,
        "dynamic_meshes": dynamic,
        "dynamic_poses": {key: value.clone().contiguous()
                          for key, value in payload["dynamic_poses"].items()},
        "ego_pose": payload["ego_pose"].clone().contiguous(),
        "metadata": {
            "schema_version": _SCHEMA_VERSION,
            "scene_name": str(payload["metadata"].get("scene_name", source.stem)),
            "coordinate_frames": {
                "static_mesh.vertices": "carla_world",
                "dynamic_meshes.vertices_local": "actor_box_local",
                "dynamic_poses.transform": "carla_world",
                "ego_pose": "carla_world",
            },
            "pose_fields": ["x", "y", "z", "roll", "pitch", "yaw"],
            "method_names": _METHOD_NAMES,
            "source_path": str(source),
            "source_fingerprint": source_fingerprint,
            "reconstruction_fingerprint": fingerprint,
            "reconstruction_config": algorithm,
            "source_metadata": payload["metadata"],
            "stats": {
                "static_vertices": len(static["vertices"]),
                "static_triangles": len(static["triangles"]),
                "static_unsupported_triangles_removed": int(
                    static.get("unsupported_triangles_removed", 0)),
                "dynamic_vertices": len(dynamic["vertices_local"]),
                "dynamic_triangles": len(dynamic["triangles"]),
                "dynamic_unsupported_triangles_removed":
                    dynamic_diagnostics["unsupported_triangles_removed"],
                "dynamic_objects": len(dynamic["actor_id"]),
                "dynamic_methods": dynamic_diagnostics["method_counts"],
                "dynamic_poisson_errors": dynamic_diagnostics["poisson_errors"],
            },
        },
    }
    check_output_payload(result)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(destination, result)
    written = torch.load(destination, map_location="cpu", weights_only=True)
    check_output_payload(written)
    if written["metadata"]["reconstruction_fingerprint"] != fingerprint:
        raise RuntimeError("Mesh 原子写入后的指纹校验失败")
    return destination


def run_reconstruction(cfg: Config, input_path=None, output_dir=None, force=False) -> dict:
    """处理单个 PT 或递归目录，保存报告并按场景恢复进度。"""
    reconstruction = cfg.mesh_reconstruction
    source = _resolve(input_path if input_path is not None else reconstruction.input_path)
    destination = _resolve(output_dir if output_dir is not None else reconstruction.output_dir)
    check_input_path(source)
    check_output_dir(destination, source, _REPO_ROOT)
    files = discover_pointclouds(source)
    destination.mkdir(parents=True, exist_ok=True)
    root = source if source.is_dir() else source.parent
    algorithm = _algorithm_config(reconstruction)
    run_fingerprint = _hash_json({
        "source": str(source), "algorithm": algorithm,
        "files": [(str(path.relative_to(root)), path.stat().st_size, path.stat().st_mtime_ns)
                  for path in files],
    })
    checkpoint_path = destination / ".mesh_checkpoint.json"
    checkpoint = _load_checkpoint(checkpoint_path, run_fingerprint, force)
    report = {"input": str(source), "output": str(destination),
              "completed": [], "skipped": [], "failed": []}
    for path in files:
        relative = path.relative_to(root)
        output = destination / relative.parent / (path.stem + ".mesh.pt")
        key = relative.as_posix()
        try:
            if not force and _existing_matches(path, output, algorithm):
                report["skipped"].append({"input": str(path), "output": str(output)})
            else:
                reconstruct_scene(path, output, cfg)
                report["completed"].append({"input": str(path), "output": str(output)})
            checkpoint["completed"][key] = str(output)
            checkpoint["errors"].pop(key, None)
        except Exception as exc:
            error = "{}: {}".format(type(exc).__name__, exc)
            checkpoint["errors"][key] = error
            report["failed"].append({"input": str(path), "output": str(output),
                                     "error": error})
        _atomic_json(checkpoint_path, checkpoint)
        _atomic_json(destination / "mesh_report.json", report)
    report["ok"] = not report["failed"]
    _atomic_json(destination / "mesh_report.json", report)
    return report


def _lidar_origins(payload):
    extrinsic = payload["metadata"].get("scene_meta", {}).get("lidar_extrinsic")
    if not isinstance(extrinsic, (list, tuple)) or len(extrinsic) < 3:
        raise ValueError("融合 metadata 缺少三维 lidar_extrinsic")
    poses = payload["ego_pose"]
    radians = torch.deg2rad(poses[:, 3:6])
    cr, cp, cy = torch.cos(radians).T
    sr, sp, sy = torch.sin(radians).T
    rotations = torch.empty((len(poses), 3, 3), dtype=torch.float32)
    rotations[:, 0, :] = torch.stack((
        cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr), dim=1)
    rotations[:, 1, :] = torch.stack((
        sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr), dim=1)
    rotations[:, 2, :] = torch.stack((sp, -cp * sr, cp * cr), dim=1)
    local = torch.tensor(extrinsic[:3], dtype=torch.float32)
    return torch.einsum("nij,j->ni", rotations, local) + poses[:, :3]


def _device(spec):
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求 CUDA，但当前 PyTorch 不支持 CUDA")
    return device


def _algorithm_config(cfg):
    values = asdict(cfg)
    values.pop("input_path")
    values.pop("output_dir")
    return values


def _source_fingerprint(path, metadata):
    stat = path.stat()
    return _hash_json({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                       "fusion_fingerprint": metadata.get("fingerprint", "")})


def _existing_matches(source, output, algorithm):
    if not output.is_file():
        return False
    try:
        source_payload = torch.load(source, map_location="cpu", weights_only=True)
        output_payload = torch.load(output, map_location="cpu", weights_only=True)
        expected = _hash_json({
            "source": _source_fingerprint(source, source_payload["metadata"]),
            "algorithm": algorithm,
        })
        return output_payload.get("metadata", {}).get("reconstruction_fingerprint") == expected
    except Exception:
        return False


def _load_checkpoint(path, fingerprint, force):
    if not force and path.is_file():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
            if current.get("run_fingerprint") == fingerprint:
                return current
        except (OSError, ValueError):
            pass
    return {"run_fingerprint": fingerprint, "completed": {}, "errors": {}}


def _atomic_torch_save(path, payload):
    temporary = path.with_name("{}.tmp.{}".format(path.name, os.getpid()))
    try:
        torch.save(payload, temporary)
        _replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("{}.tmp.{}".format(path.name, os.getpid()))
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _replace(source, destination):
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt + 1 == _REPLACE_ATTEMPTS:
                raise
            time.sleep(_REPLACE_DELAY_S)


def _hash_json(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve(path):
    value = Path(path)
    return value.resolve() if value.is_absolute() else (_REPO_ROOT / value).resolve()


def _remove_empty_dir(path):
    try:
        Path(path).rmdir()
    except OSError:
        pass
