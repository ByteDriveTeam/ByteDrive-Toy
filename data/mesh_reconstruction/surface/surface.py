"""以 PyTorch 预处理和 Open3D Poisson 生成三角 Mesh，并可选执行水密修复。

模块: data/mesh_reconstruction/surface/surface.py
依赖: dataclasses, itertools, os, pathlib, subprocess, sys, uuid, numpy, open3d, torch,
      data.mesh_reconstruction.surface.checks
读取配置: mesh_reconstruction.static/dynamic 的 voxel_size_m/normal_radius_m/
          normal_max_nn/depth/scale/max_triangles/tag_radius_m/
          prune_unsupported_surfaces/support_radius_m
对外接口:
    - reconstruct_surface(points, tags, orientation_targets, cfg, hole_radius_m,
                          enable_watertight_repair, device, batch_size,
                          candidate_budget, poisson_threads) -> dict
    - reconstruct_surface_isolated(points, tags, orientation_targets, cfg, hole_radius_m,
                                   enable_watertight_repair, device, batch_size,
                                   candidate_budget, poisson_threads, work_dir) -> dict
说明: Open3D 负责 CPU 法线邻域与 Poisson；默认仅清理退化/重复面，保留原始开放表面。
      开启可选水密修复后，才执行非流形边剔除、闭环补洞及强制拓扑验收。
"""

from dataclasses import asdict
from itertools import product
import os
from pathlib import Path
import subprocess
import sys
import uuid

import numpy as np
import open3d as o3d
import torch

from data.mesh_reconstruction.surface.checks.surface_checks import (
    check_sampled_points,
    check_surface_inputs,
    check_surface_output,
)

__all__ = ["reconstruct_surface", "reconstruct_surface_isolated"]

_REPO_ROOT = Path(__file__).resolve().parents[3]


def reconstruct_surface_isolated(points, tags, orientation_targets, cfg, hole_radius_m,
                                 enable_watertight_repair, device, batch_size,
                                 candidate_budget, poisson_threads, work_dir):
    """在隔离子进程执行原生 Poisson，使崩溃或超时不能终止批处理主进程。"""
    work = Path(work_dir).resolve()
    assert work != _REPO_ROOT and _REPO_ROOT in work.parents, \
        "Poisson 临时目录必须严格位于项目目录内: {}".format(work)
    work.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    request, response = work / (token + ".request.pt"), work / (token + ".response.pt")
    torch.save({
        "points": points.cpu(), "tags": tags.cpu(),
        "orientation_targets": orientation_targets.cpu() if orientation_targets is not None else None,
        "cfg": asdict(cfg), "hole_radius_m": hole_radius_m,
        "enable_watertight_repair": enable_watertight_repair, "device": str(device),
        "batch_size": batch_size, "candidate_budget": candidate_budget,
        "poisson_threads": poisson_threads,
    }, request)
    command = [sys.executable, "-m", "data.mesh_reconstruction.surface.worker",
               str(request), str(response)]
    environment = os.environ.copy()
    if poisson_threads > 0:
        environment["OMP_NUM_THREADS"] = str(poisson_threads)
    try:
        completed = subprocess.run(
            command, cwd=str(_REPO_ROOT), env=environment,
            capture_output=True, text=True, timeout=cfg.timeout_s, check=False)
        if completed.returncode != 0 or not response.is_file():
            streams = "\n".join(value for value in (
                completed.stdout.strip(), completed.stderr.strip()) if value)
            detail = streams or "原生进程返回码 {}".format(completed.returncode)
            raise RuntimeError("隔离 Poisson 失败: {}".format(detail[-2000:]))
        result = torch.load(response, map_location="cpu", weights_only=True)
        check_surface_output(result)
        return result
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("隔离 Poisson 超过 {:.1f} 秒".format(cfg.timeout_s)) from exc
    finally:
        request.unlink(missing_ok=True)
        response.unlink(missing_ok=True)


def reconstruct_surface(points, tags, orientation_targets, cfg, hole_radius_m,
                        enable_watertight_repair, device, batch_size,
                        candidate_budget, poisson_threads):
    """从带标签点云重建表面，并按配置选择是否强制修复为水密 Mesh。

    参数:
        points:              float32[N,3] 世界或对象局部点
        tags:                uint8[N] 语义标签
        orientation_targets: 静态点对应的候选 LiDAR 世界原点；None 表示法线背离局部原点
        cfg:                 静态或动态 Poisson 配置
        hole_radius_m:       水密修复开启时的补洞最大近似半径
        enable_watertight_repair: 是否补洞并强制水密拓扑
        device:              张量预处理设备
        batch_size:          最近目标与标签查询分批大小
        candidate_budget:    单批语义空间哈希候选总数上限
        poisson_threads:     Open3D Poisson 线程数
    返回:
        CPU torch.Tensor 组成的 Mesh 字典，包含真实水密状态
    """
    check_surface_inputs(points, tags, orientation_targets)
    sampled_points, sampled_tags = _voxel_downsample(
        points.to(device), tags.to(device), cfg.voxel_size_m)
    if len(sampled_points) < 4:
        raise ValueError("体素下采样后不足四个点，无法重建三维表面")
    check_sampled_points(sampled_points)
    _progress("体素聚合", len(sampled_points))
    cloud = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(sampled_points.cpu().numpy().astype(np.float64)))
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=cfg.normal_radius_m, max_nn=cfg.normal_max_nn))
    _progress("法线估计", len(sampled_points))
    normals = torch.from_numpy(np.asarray(cloud.normals).copy()).to(
        device=device, dtype=torch.float32)
    oriented = _orient_normals(
        sampled_points, normals,
        orientation_targets.to(device) if orientation_targets is not None else None,
        batch_size)
    cloud.normals = o3d.utility.Vector3dVector(oriented.cpu().numpy().astype(np.float64))
    _progress("法线定向", len(sampled_points))
    raw, _densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud, depth=cfg.depth, scale=cfg.scale, n_threads=poisson_threads)
    _progress("Poisson", len(raw.triangles))
    vertices, triangles = (_repair(raw, hole_radius_m)
                           if enable_watertight_repair else _clean_tensors(raw))
    _progress("拓扑修复" if enable_watertight_repair else "拓扑清理", len(triangles))
    is_watertight = _valid_topology(vertices, triangles)
    if enable_watertight_repair and not is_watertight:
        raise RuntimeError("Poisson 结果经补洞后仍未形成水密流形")
    if len(triangles) > cfg.max_triangles:
        simplified = _legacy_mesh(vertices, triangles).simplify_quadric_decimation(
            cfg.max_triangles)
        candidate_vertices, candidate_triangles = (
            _repair(simplified, hole_radius_m)
            if enable_watertight_repair else _clean_tensors(simplified))
        candidate_watertight = _valid_topology(candidate_vertices, candidate_triangles)
        if not enable_watertight_repair or candidate_watertight:
            vertices, triangles = candidate_vertices, candidate_triangles
            is_watertight = candidate_watertight
    vertices = vertices.to(torch.float32)
    vertex_tags, supported = _nearest_tags(
        vertices.to(device), sampled_points, sampled_tags,
        cfg.voxel_size_m, cfg.tag_radius_m, cfg.support_radius_m,
        batch_size, candidate_budget)
    vertex_tags, supported = vertex_tags.cpu(), supported.cpu()
    removed_triangles = 0
    if cfg.prune_unsupported_surfaces and not enable_watertight_repair:
        before = len(triangles)
        vertices, triangles, vertex_tags = _prune_unsupported(
            vertices, triangles, vertex_tags, supported)
        removed_triangles = before - len(triangles)
        if not len(triangles):
            raise RuntimeError("观测支撑裁剪移除了全部 Poisson 三角形")
        is_watertight = _valid_topology(vertices, triangles)
        _progress("无支撑表面裁剪", removed_triangles)
    vertex_normals = _vertex_normals(vertices, triangles)
    _progress("标签迁移", len(vertices))
    result = {
        "vertices": vertices.contiguous(),
        "triangles": triangles.contiguous(),
        "vertex_normals": vertex_normals.contiguous(),
        "vertex_obj_tag": vertex_tags.to(torch.uint8).contiguous(),
        "is_watertight": torch.tensor(is_watertight, dtype=torch.bool),
        "unsupported_triangles_removed": torch.tensor(
            removed_triangles, dtype=torch.int64),
    }
    check_surface_output(result)
    return result


def _voxel_downsample(points, tags, voxel_size):
    cells = torch.floor(points / voxel_size).to(torch.int64)
    _unique, inverse = torch.unique(cells, dim=0, return_inverse=True)
    count = torch.bincount(inverse, minlength=len(_unique))
    sums = torch.zeros((len(_unique), 3), dtype=torch.float32, device=points.device)
    sums.scatter_add_(0, inverse[:, None].expand(-1, 3), points)
    sampled = sums / count[:, None]
    pairs, pair_count = torch.unique(
        inverse.to(torch.int64) * 256 + tags.to(torch.int64), return_counts=True)
    pair_voxel, pair_tag = pairs // 256, pairs % 256
    maximum = torch.zeros(len(_unique), dtype=pair_count.dtype, device=points.device)
    maximum.scatter_reduce_(0, pair_voxel, pair_count, reduce="amax", include_self=True)
    best = pair_count == maximum[pair_voxel]
    sampled_tags = torch.full(
        (len(_unique),), 255, dtype=torch.int64, device=points.device)
    sampled_tags.scatter_reduce_(
        0, pair_voxel[best], pair_tag[best], reduce="amin", include_self=True)
    return sampled, sampled_tags.to(torch.uint8)


def _orient_normals(points, normals, targets, batch_size):
    if targets is None:
        reference = points
    else:
        references = []
        for start in range(0, len(points), batch_size):
            chunk = points[start:start + batch_size]
            nearest = torch.cdist(chunk, targets).argmin(dim=1)
            references.append(targets[nearest] - chunk)
        reference = torch.cat(references)
    flip = (normals * reference).sum(dim=1) < 0
    return torch.where(flip[:, None], -normals, normals)


def _repair(mesh, hole_radius):
    vertices, triangles = _clean_tensors(mesh)
    closed = _closed_edges(triangles)
    _progress("边闭合检查", int(closed))
    if not closed:
        legacy = _legacy_mesh(vertices, triangles)
        legacy.remove_non_manifold_edges()
        vertices, triangles = _clean_tensors(legacy)
        _progress("非流形清理", len(triangles))
        _progress("孔洞填补输入", len(triangles))
        vertices, triangles = _fill_boundary_loops(
            vertices, triangles, hole_radius)
        _progress("孔洞填补输出", len(triangles))
    return vertices, triangles


def _clean_tensors(mesh):
    raw_vertices = np.asarray(mesh.vertices).copy()
    unique_vertices, vertex_inverse = np.unique(raw_vertices, axis=0, return_inverse=True)
    vertices = torch.from_numpy(unique_vertices).to(torch.float64)
    raw_triangles = np.asarray(mesh.triangles).copy()
    triangles = torch.from_numpy(vertex_inverse[raw_triangles]).to(torch.int64)
    _progress("拓扑张量化", len(triangles))
    if not len(vertices) or not len(triangles):
        return vertices, triangles
    faces = vertices[triangles]
    distinct = (triangles[:, 0] != triangles[:, 1]) \
        & (triangles[:, 1] != triangles[:, 2]) \
        & (triangles[:, 2] != triangles[:, 0])
    area = torch.linalg.vector_norm(
        torch.linalg.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0]), dim=1)
    triangles = triangles[distinct & (area > torch.finfo(vertices.dtype).eps)]
    _progress("退化面清理", len(triangles))
    canonical = torch.sort(triangles, dim=1).values
    unique_faces, inverse = torch.unique(canonical, dim=0, return_inverse=True)
    first = torch.full((len(unique_faces),), len(triangles), dtype=torch.int64)
    first.scatter_reduce_(0, inverse, torch.arange(len(triangles)),
                          reduce="amin", include_self=True)
    triangles = triangles[first]
    _progress("重复面清理", len(triangles))
    used, compact = torch.unique(triangles.flatten(), sorted=True, return_inverse=True)
    _progress("顶点压紧", len(used))
    return vertices[used].contiguous(), compact.reshape(-1, 3).contiguous()


def _valid_topology(vertices, triangles):
    return len(vertices) > 0 and len(triangles) > 0 \
        and bool(torch.isfinite(vertices).all()) and _closed_edges(triangles)


def _closed_edges(triangles):
    if not len(triangles):
        return False
    edges = torch.cat((triangles[:, (0, 1)], triangles[:, (1, 2)],
                       triangles[:, (2, 0)]))
    _edges, counts = torch.unique(torch.sort(edges, dim=1).values,
                                  dim=0, return_counts=True)
    _progress("边界边", int((counts == 1).sum()))
    _progress("非流形边", int((counts > 2).sum()))
    return bool((counts == 2).all())


def _fill_boundary_loops(vertices, triangles, hole_radius):
    directed = torch.cat((triangles[:, (0, 1)], triangles[:, (1, 2)],
                          triangles[:, (2, 0)]))
    canonical = torch.sort(directed, dim=1).values
    _unique, inverse, counts = torch.unique(
        canonical, dim=0, return_inverse=True, return_counts=True)
    boundary = directed[counts[inverse] == 1]
    if not len(boundary):
        return vertices, triangles
    edge_list = [(int(start), int(end)) for start, end in boundary.tolist()]
    boundary_vertices, degrees = torch.unique(boundary.flatten(), return_counts=True)
    odd = boundary_vertices[degrees % 2 != 0]
    if len(odd):
        raise RuntimeError("非流形清理后仍有 {} 个奇数度边界顶点".format(len(odd)))
    incident = {}
    for edge_index, (start, end) in enumerate(edge_list):
        incident.setdefault(start, []).append(edge_index)
        incident.setdefault(end, []).append(edge_index)
    for indices in incident.values():
        indices.sort(reverse=True)
    unused, circuits = set(range(len(edge_list))), []
    while unused:
        start_edge = min(unused)
        stack, circuit = [edge_list[start_edge][0]], []
        while stack:
            current = stack[-1]
            candidates = incident.get(current, [])
            while candidates and candidates[-1] not in unused:
                candidates.pop()
            if candidates:
                edge_index = candidates.pop()
                unused.remove(edge_index)
                first, second = edge_list[edge_index]
                stack.append(second if current == first else first)
            else:
                circuit.append(stack.pop())
        circuit.reverse()
        if len(circuit) < 4 or circuit[0] != circuit[-1]:
            raise RuntimeError("边界未形成长度至少为三的有向闭环")
        circuits.extend(_split_simple_cycles(circuit))
    loops = [circuit[:-1] for circuit in circuits]
    centers, caps = [], []
    for loop in loops:
        loop_tensor = torch.tensor(loop, dtype=torch.int64)
        center = vertices[loop_tensor].mean(dim=0)
        radius = torch.linalg.vector_norm(vertices[loop_tensor] - center, dim=1).max()
        if float(radius) > hole_radius:
            raise RuntimeError("边界孔洞半径 {:.3f}m 超过配置上限".format(float(radius)))
        center_index = len(vertices) + len(centers)
        centers.append(center)
        caps.extend((loop[(index + 1) % len(loop)], loop[index], center_index)
                    for index in range(len(loop)))
    return (torch.cat((vertices, torch.stack(centers))).contiguous(),
            torch.cat((triangles, torch.tensor(caps, dtype=torch.int64))).contiguous())


def _split_simple_cycles(circuit):
    positions = {}
    for index, vertex in enumerate(circuit[:-1]):
        if vertex in positions:
            first = positions[vertex]
            inner = circuit[first:index + 1]
            outer = circuit[:first + 1] + circuit[index + 1:]
            return _split_simple_cycles(inner) + _split_simple_cycles(outer)
        positions[vertex] = index
    return [circuit]


def _legacy_mesh(vertices, triangles):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.numpy().astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(triangles.numpy().astype(np.int32))
    return mesh


def _vertex_normals(vertices, triangles):
    faces = vertices[triangles]
    face_normals = torch.linalg.cross(
        faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    normals = torch.zeros_like(vertices)
    normals.index_add_(
        0, triangles.flatten(), face_normals[:, None, :].expand(-1, 3, -1).reshape(-1, 3))
    return torch.nn.functional.normalize(normals, dim=1)


def _nearest_tags(vertices, points, tags, voxel_size, tag_radius, support_radius,
                  batch_size, candidate_budget):
    cells = torch.floor(points / voxel_size).to(torch.int64)
    lower = cells.min(dim=0).values
    upper = cells.max(dim=0).values
    shape = upper - lower + 1
    keys = _linear_keys(cells, lower, shape)
    keys, order = torch.sort(keys)
    sorted_points, sorted_tags = points[order], tags[order]
    reach = int(np.ceil(max(tag_radius, support_radius) / voxel_size))
    offsets = torch.tensor(
        list(product(range(-reach, reach + 1), repeat=3)),
        dtype=torch.int64, device=vertices.device)
    chunk_size = max(1, min(batch_size, candidate_budget // len(offsets)))
    output, support = [], []
    for start in range(0, len(vertices), chunk_size):
        chunk = vertices[start:start + chunk_size]
        query_cells = torch.floor(chunk / voxel_size).to(torch.int64)
        candidates = query_cells[:, None, :] + offsets[None, :, :]
        valid_cells = ((candidates >= lower) & (candidates <= upper)).all(dim=2)
        query_keys = _linear_keys(candidates, lower, shape)
        positions = torch.searchsorted(keys, query_keys)
        safe = positions.clamp(max=max(len(keys) - 1, 0))
        matches = valid_cells & (positions < len(keys)) & (keys[safe] == query_keys)
        candidate_points = sorted_points[safe]
        distances = torch.linalg.vector_norm(candidate_points - chunk[:, None, :], dim=2)
        distances = torch.where(matches, distances, torch.inf)
        best_distance, best_position = distances.min(dim=1)
        best_tag = sorted_tags[safe.gather(1, best_position[:, None]).squeeze(1)]
        output.append(torch.where(best_distance <= tag_radius, best_tag, 0).to(torch.uint8))
        support.append(best_distance <= support_radius)
    return torch.cat(output), torch.cat(support)


def _prune_unsupported(vertices, triangles, tags, supported):
    keep = supported[triangles].all(dim=1)
    triangles = triangles[keep]
    if not len(triangles):
        return vertices[:0], triangles, tags[:0]
    used, inverse = torch.unique(triangles.flatten(), sorted=True, return_inverse=True)
    return (vertices[used].contiguous(), inverse.reshape(-1, 3).contiguous(),
            tags[used].contiguous())


def _linear_keys(cells, lower, shape):
    shifted = cells - lower
    return (shifted[..., 0] * shape[1] + shifted[..., 1]) * shape[2] + shifted[..., 2]


def _progress(stage, count):
    if os.environ.get("BYTEDRIVE_MESH_WORKER") == "1":
        print("[MeshWorker] {}: {:,}".format(stage, count), flush=True)
