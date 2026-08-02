"""Open3D Mesh 渲染：静态表面着色、动态逐帧刚体放置与 actor 轨迹。

模块: vis/reconstructed_mesh_vis/render/render.py
依赖: dataclasses, numpy, open3d, torch, vis.reconstructed_mesh_vis.render.checks
读取配置: reconstructed_mesh_vis.static_rgb/dynamic_rgb/unknown_rgb/poisson_rgb/
          reuse_rgb/box_rgb；model.driving.bev.x/y_min/max_m
对外接口:
    - RenderState
    - render_static_mesh(data, state, cfg, bev_cfg) -> open3d.geometry.TriangleMesh
    - render_dynamic_mesh(data, state, cfg, bev_cfg) -> open3d.geometry.TriangleMesh
    - render_trajectories(data) -> open3d.geometry.LineSet
"""

from dataclasses import dataclass

import numpy as np
import open3d as o3d
import torch

from vis.reconstructed_mesh_vis.render.checks.render_checks import check_render_state
from vis.reconstructed_pointcloud_vis.render import current_bev_mask

__all__ = ["RenderState", "render_static_mesh", "render_dynamic_mesh",
           "render_trajectories"]

_SEMANTIC_RGB = torch.tensor((
    (0, 0, 0), (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
    (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
    (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
    (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100),
    (0, 0, 230), (119, 11, 32), (110, 110, 110), (255, 190, 0), (81, 0, 81),
    (45, 60, 150), (255, 255, 255), (81, 0, 81), (150, 100, 100),
    (230, 150, 140), (180, 165, 180)), dtype=torch.float64) / 255


@dataclass
class RenderState:
    """描述当前 Mesh 帧、图层、着色和相机跟随状态。"""

    show_static: bool
    show_dynamic: bool
    show_trajectory: bool
    follow_ego: bool
    frame_index: int
    color_mode: str


def render_static_mesh(data, state, cfg, bev_cfg):
    """构造静态世界 Mesh；跟随自车时只保留当前 BEV 内三角形。"""
    check_render_state(data, state)
    if not state.show_static:
        return o3d.geometry.TriangleMesh()
    source = data.static
    vertices, triangles = source["vertices"], source["triangles"]
    normals, tags = source["vertex_normals"], source["vertex_obj_tag"]
    if state.follow_ego:
        vertices, triangles, (normals, tags) = _crop_bev_mesh(
            vertices, triangles, (normals, tags),
            data.ego_pose[state.frame_index], bev_cfg)
    colors = _colors(tags, None, None, state.color_mode, cfg, False)
    return _legacy_mesh(vertices, triangles, normals, colors)


def render_dynamic_mesh(data, state, cfg, bev_cfg):
    """放置当前帧动态 Mesh；跟随自车时只保留当前 BEV 内三角形。"""
    check_render_state(data, state)
    if not state.show_dynamic:
        return o3d.geometry.TriangleMesh()
    poses = data.poses
    pose_indices = torch.nonzero(
        poses["frame_index"] == state.frame_index, as_tuple=False).flatten()
    objects = poses["object_index"][pose_indices]
    vertex_offsets, triangle_offsets = (
        data.dynamic["vertex_offsets"], data.dynamic["triangle_offsets"])
    visible = vertex_offsets[objects + 1] > vertex_offsets[objects]
    pose_indices, objects = pose_indices[visible], objects[visible]
    if not len(objects):
        return o3d.geometry.TriangleMesh()
    vertex_chunks, normal_chunks, tag_chunks = [], [], []
    actor_chunks, method_chunks, triangle_chunks = [], [], []
    output_offset = 0
    for object_index in objects.tolist():
        first, last = int(vertex_offsets[object_index]), int(vertex_offsets[object_index + 1])
        triangle_first = int(triangle_offsets[object_index])
        triangle_last = int(triangle_offsets[object_index + 1])
        count = last - first
        vertex_chunks.append(data.dynamic["vertices_local"][first:last])
        normal_chunks.append(data.dynamic["vertex_normals_local"][first:last])
        tag_chunks.append(data.dynamic["vertex_obj_tag"][first:last])
        actor_chunks.append(data.dynamic["actor_id"][object_index].repeat(count))
        method_chunks.append(data.dynamic["method_code"][object_index].repeat(count))
        triangle_chunks.append(
            data.dynamic["triangles"][triangle_first:triangle_last] - first + output_offset)
        output_offset += count
    counts = vertex_offsets[objects + 1] - vertex_offsets[objects]
    vertex_pose = torch.repeat_interleave(torch.arange(len(objects)), counts)
    matrices = _pose_matrices(poses["transform"][pose_indices])
    local_vertices, local_normals = torch.cat(vertex_chunks), torch.cat(normal_chunks)
    rotations = matrices[vertex_pose, :3, :3]
    vertices = torch.einsum("nij,nj->ni", rotations, local_vertices) \
        + matrices[vertex_pose, :3, 3]
    normals = torch.einsum("nij,nj->ni", rotations, local_normals)
    triangles = torch.cat(triangle_chunks)
    tags, actors, methods = torch.cat(tag_chunks), torch.cat(actor_chunks), torch.cat(method_chunks)
    if state.follow_ego:
        vertices, triangles, (normals, tags, actors, methods) = _crop_bev_mesh(
            vertices, triangles, (normals, tags, actors, methods),
            data.ego_pose[state.frame_index], bev_cfg)
    colors = _colors(tags, actors, methods, state.color_mode, cfg, True)
    return _legacy_mesh(vertices, triangles, normals, colors)


def _crop_bev_mesh(vertices, triangles, attributes, ego_pose, bev_cfg):
    inside = torch.from_numpy(current_bev_mask(
        vertices.numpy(), ego_pose.numpy(), bev_cfg))
    kept = triangles[inside[triangles].all(dim=1)]
    if not len(kept):
        return vertices[:0], triangles[:0], tuple(value[:0] for value in attributes)
    used, inverse = torch.unique(kept.flatten(), sorted=True, return_inverse=True)
    return (vertices[used], inverse.reshape(-1, 3),
            tuple(value[used] for value in attributes))


def render_trajectories(data):
    """由全部动态对象位姿生成世界坐标轨迹，包含无 Mesh actor。"""
    poses = data.poses
    geometry = o3d.geometry.LineSet()
    if not len(poses["object_index"]):
        return geometry
    keys = poses["object_index"].to(torch.int64) * data.num_frames \
        + poses["frame_index"].to(torch.int64)
    order = torch.argsort(keys)
    object_index = poses["object_index"][order]
    centers = poses["transform"][order, :3]
    connected = object_index[1:] == object_index[:-1]
    starts = torch.nonzero(connected, as_tuple=False).flatten()
    lines = torch.stack((starts, starts + 1), dim=1).to(torch.int32)
    actor_id = data.dynamic["actor_id"][object_index[starts]]
    geometry.points = o3d.utility.Vector3dVector(centers.numpy().astype(np.float64))
    geometry.lines = o3d.utility.Vector2iVector(lines.numpy())
    geometry.colors = o3d.utility.Vector3dVector(_actor_colors(actor_id).numpy())
    return geometry


def _legacy_mesh(vertices, triangles, normals, colors):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.numpy().astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(triangles.numpy().astype(np.int32))
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals.numpy().astype(np.float64))
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors.numpy().astype(np.float64))
    return mesh


def _colors(tags, actors, methods, mode, cfg, dynamic):
    if mode == "semantic":
        colors = torch.as_tensor(cfg.unknown_rgb, dtype=torch.float64).repeat(len(tags), 1) / 255
        known = tags.to(torch.int64) < len(_SEMANTIC_RGB)
        colors[known] = _SEMANTIC_RGB[tags[known].to(torch.int64)]
        return colors
    if mode == "source":
        rgb = cfg.dynamic_rgb if dynamic else cfg.static_rgb
        return torch.as_tensor(rgb, dtype=torch.float64).repeat(len(tags), 1) / 255
    if mode == "actor":
        return _actor_colors(actors) if dynamic else \
            torch.as_tensor(cfg.static_rgb, dtype=torch.float64).repeat(len(tags), 1) / 255
    if not dynamic:
        return torch.as_tensor(cfg.poisson_rgb, dtype=torch.float64).repeat(len(tags), 1) / 255
    palette = torch.as_tensor((cfg.unknown_rgb, cfg.poisson_rgb, cfg.reuse_rgb,
                               cfg.reuse_rgb, cfg.box_rgb), dtype=torch.float64) / 255
    return palette[methods.to(torch.int64)]


def _actor_colors(actor_ids):
    ids = actor_ids.to(torch.float64)
    hue = torch.remainder(ids * 0.6180339887498949, 1)
    sector = torch.floor(hue * 6).to(torch.int64)
    fraction = hue * 6 - sector
    value, minimum = torch.ones_like(hue), torch.full_like(hue, 0.22)
    rising = minimum + (value - minimum) * fraction
    falling = value - (value - minimum) * fraction
    table = torch.stack((
        torch.stack((value, rising, minimum), dim=1),
        torch.stack((falling, value, minimum), dim=1),
        torch.stack((minimum, value, rising), dim=1),
        torch.stack((minimum, falling, value), dim=1),
        torch.stack((rising, minimum, value), dim=1),
        torch.stack((value, minimum, falling), dim=1)))
    return table[sector % 6, torch.arange(len(ids))]


def _pose_matrices(poses):
    radians = torch.deg2rad(poses[:, 3:6])
    cr, cp, cy = torch.cos(radians).T
    sr, sp, sy = torch.sin(radians).T
    matrices = torch.eye(4, dtype=torch.float32).repeat(len(poses), 1, 1)
    matrices[:, :3, 3] = poses[:, :3]
    matrices[:, 0, :3] = torch.stack((
        cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr), dim=1)
    matrices[:, 1, :3] = torch.stack((
        sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr), dim=1)
    matrices[:, 2, :3] = torch.stack((sp, -cp * sr, cp * cr), dim=1)
    return matrices
