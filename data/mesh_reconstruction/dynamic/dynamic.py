"""按点覆盖率重建动态对象，并依次执行相同 donor、相似 donor 与 Box 回退。

模块: data/mesh_reconstruction/dynamic/dynamic.py
依赖: copy, torch, data.mesh_reconstruction.surface,
      data.mesh_reconstruction.dynamic.checks
读取配置: mesh_reconstruction.dynamic 全树；mesh_reconstruction.repair；
          mesh_reconstruction.device/tensor_batch_size/poisson_threads
对外接口:
    - reconstruct_dynamic_objects(objects, source_voxel_size, cfg, device,
                                  work_dir) -> tuple[dict,dict]
说明: donor 只来自原始点充足且自身 Poisson 成功的对象，复用结果不再成为 donor。
"""

import torch

from data.mesh_reconstruction.dynamic.checks.dynamic_checks import (
    check_dynamic_inputs,
    check_dynamic_output,
)
from data.mesh_reconstruction.surface import reconstruct_surface_isolated

__all__ = ["reconstruct_dynamic_objects"]

_UNOBSERVED = 0
_POISSON = 1
_EXACT_REUSE = 2
_SIMILAR_REUSE = 3
_BOX = 4


def reconstruct_dynamic_objects(objects, source_voxel_size, cfg, device, work_dir):
    """重建全部动态对象并返回 packed Mesh 与诊断统计。"""
    check_dynamic_inputs(objects, source_voxel_size)
    dynamic = cfg.dynamic
    offsets = objects["point_offsets"]
    counts = offsets[1:] - offsets[:-1]
    extents = objects["extent"]
    area = 8 * (extents[:, 0] * extents[:, 1]
                + extents[:, 0] * extents[:, 2]
                + extents[:, 1] * extents[:, 2])
    coverage = counts.to(torch.float32) * source_voxel_size ** 2 / area.clamp_min(1e-8)
    dense = (counts >= dynamic.min_points) & (coverage >= dynamic.min_surface_coverage)
    meshes = [None] * len(counts)
    methods = torch.zeros(len(counts), dtype=torch.uint8)
    donors = torch.full((len(counts),), -1, dtype=torch.int64)
    errors = {}
    for index in torch.nonzero(dense, as_tuple=False).flatten().tolist():
        start, end = int(offsets[index]), int(offsets[index + 1])
        try:
            meshes[index] = reconstruct_surface_isolated(
                objects["xyz_local"][start:end], objects["obj_tag"][start:end], None,
                dynamic, cfg.repair.dynamic_hole_radius_m, cfg.repair.enabled, device,
                cfg.tensor_batch_size, cfg.tag_candidate_budget, cfg.poisson_threads,
                work_dir)
            methods[index] = _POISSON
        except Exception as exc:
            errors[str(int(objects["actor_id"][index]))] = "{}: {}".format(
                type(exc).__name__, exc)
    donor_indices = [index for index in range(len(meshes))
                     if meshes[index] is not None and bool(dense[index])]
    for index in range(len(meshes)):
        if meshes[index] is not None or int(counts[index]) == 0:
            continue
        donor, method = _select_donor(
            index, donor_indices, objects, coverage, dynamic)
        if donor is not None:
            meshes[index] = _reuse_mesh(
                meshes[donor], extents[index], extents[donor], method == _SIMILAR_REUSE)
            methods[index], donors[index] = method, donor
        else:
            meshes[index] = _box_mesh(extents[index])
            methods[index] = _BOX
    packed = _pack(objects, meshes, methods, donors)
    packed["source_point_count"] = counts.to(torch.int64).contiguous()
    packed["source_surface_coverage"] = coverage.to(torch.float32).contiguous()
    diagnostics = {
        "point_count": counts.to(torch.int64).contiguous(),
        "surface_coverage": coverage.to(torch.float32).contiguous(),
        "poisson_errors": errors,
        "unsupported_triangles_removed": int(
            packed["unsupported_triangles_removed"].sum()),
        "method_counts": {
            name: int((methods == code).sum()) for code, name in (
                (_UNOBSERVED, "unobserved"), (_POISSON, "poisson"),
                (_EXACT_REUSE, "exact_reuse"), (_SIMILAR_REUSE, "similar_reuse"),
                (_BOX, "box"))
        },
    }
    check_dynamic_output(packed)
    return packed, diagnostics


def _select_donor(index, donor_indices, objects, coverage, cfg):
    if not donor_indices:
        return None, None
    extent = objects["extent"][index]
    class_id = int(objects["class_id"][index])
    same_class = [candidate for candidate in donor_indices
                  if int(objects["class_id"][candidate]) == class_id]
    exact = [candidate for candidate in same_class if bool(
        (torch.abs(objects["extent"][candidate] - extent)
         <= cfg.exact_extent_tolerance_m).all())]
    if exact:
        return _rank(exact, extent, objects, coverage), _EXACT_REUSE
    similar = [candidate for candidate in same_class if bool(
        (torch.abs(objects["extent"][candidate] - extent)
         / torch.maximum(objects["extent"][candidate], extent).clamp_min(1e-8)
         <= cfg.similar_extent_relative_tolerance).all())]
    return (_rank(similar, extent, objects, coverage), _SIMILAR_REUSE) \
        if similar else (None, None)


def _rank(candidates, target_extent, objects, coverage):
    return min(candidates, key=lambda candidate: (
        float(torch.linalg.vector_norm(
            (objects["extent"][candidate] - target_extent)
            / torch.maximum(objects["extent"][candidate], target_extent).clamp_min(1e-8))),
        -float(coverage[candidate]), int(objects["actor_id"][candidate])))


def _reuse_mesh(mesh, target_extent, donor_extent, scale):
    result = {key: value.clone() for key, value in mesh.items()}
    result["unsupported_triangles_removed"] = torch.tensor(0, dtype=torch.int64)
    if not scale:
        return result
    factors = target_extent / donor_extent.clamp_min(1e-8)
    result["vertices"] *= factors
    normals = result["vertex_normals"] / factors
    result["vertex_normals"] = torch.nn.functional.normalize(normals, dim=1)
    return result


def _box_mesh(extent):
    signs = torch.tensor((
        (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
        (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)), dtype=torch.float32)
    triangles = torch.tensor((
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)), dtype=torch.int64)
    vertices = signs * extent.to(torch.float32)
    return {
        "vertices": vertices,
        "triangles": triangles,
        "vertex_normals": _vertex_normals(vertices, triangles),
        "vertex_obj_tag": torch.zeros(8, dtype=torch.uint8),
        "is_watertight": torch.tensor(True, dtype=torch.bool),
        "unsupported_triangles_removed": torch.tensor(0, dtype=torch.int64),
    }


def _vertex_normals(vertices, triangles):
    faces = vertices[triangles]
    normals = torch.linalg.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    output = torch.zeros_like(vertices)
    output.index_add_(0, triangles.flatten(), normals[:, None, :].expand(-1, 3, -1).reshape(-1, 3))
    return torch.nn.functional.normalize(output, dim=1)


def _pack(objects, meshes, methods, donors):
    vertex_counts = torch.tensor(
        [len(mesh["vertices"]) if mesh is not None else 0 for mesh in meshes],
        dtype=torch.int64)
    triangle_counts = torch.tensor(
        [len(mesh["triangles"]) if mesh is not None else 0 for mesh in meshes],
        dtype=torch.int64)
    vertex_offsets = torch.cat((torch.zeros(1, dtype=torch.int64), vertex_counts.cumsum(0)))
    triangle_offsets = torch.cat((torch.zeros(1, dtype=torch.int64), triangle_counts.cumsum(0)))
    present = [(index, mesh) for index, mesh in enumerate(meshes) if mesh is not None]
    vertices = _cat([mesh["vertices"] for _, mesh in present], (0, 3), torch.float32)
    normals = _cat([mesh["vertex_normals"] for _, mesh in present], (0, 3), torch.float32)
    tags = _cat([mesh["vertex_obj_tag"] for _, mesh in present], (0,), torch.uint8)
    triangles = _cat([
        mesh["triangles"] + vertex_offsets[index] for index, mesh in present
    ], (0, 3), torch.int64)
    return {
        "actor_id": objects["actor_id"].clone().contiguous(),
        "class_id": objects["class_id"].clone().contiguous(),
        "extent": objects["extent"].clone().contiguous(),
        "vertex_offsets": vertex_offsets.contiguous(),
        "triangle_offsets": triangle_offsets.contiguous(),
        "vertices_local": vertices.contiguous(),
        "triangles": triangles.contiguous(),
        "vertex_normals_local": normals.contiguous(),
        "vertex_obj_tag": tags.contiguous(),
        "method_code": methods.contiguous(),
        "donor_object_index": donors.contiguous(),
        "is_watertight": torch.tensor([
            bool(mesh["is_watertight"]) if mesh is not None else False
            for mesh in meshes
        ], dtype=torch.bool),
        "unsupported_triangles_removed": torch.tensor([
            int(mesh.get("unsupported_triangles_removed", 0))
            if mesh is not None else 0 for mesh in meshes
        ], dtype=torch.int64),
    }


def _cat(values, shape, dtype):
    return torch.cat(values) if values else torch.empty(shape, dtype=dtype)
