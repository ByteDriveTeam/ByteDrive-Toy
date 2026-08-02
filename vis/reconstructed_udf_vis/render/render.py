"""把静态世界和当前帧动态局部 TUDF 渲染为 Open3D 稀疏点体素。

模块: vis/reconstructed_udf_vis/render/render.py
依赖: dataclasses, numpy, open3d, torch, reconstructed_mesh_vis,
      reconstructed_pointcloud_vis, reconstructed_udf_vis.render.checks
读取配置: reconstructed_udf_vis 的体素上限/颜色；model.driving.bev
对外接口:
    - RenderState
    - render_voxels(data, state, cfg, bev_cfg) -> open3d.geometry.PointCloud
    - render_trajectories(data) -> open3d.geometry.LineSet
"""

from dataclasses import dataclass

import numpy as np
import open3d as o3d
import torch

from vis.reconstructed_mesh_vis.render import render_trajectories
from vis.reconstructed_pointcloud_vis.render import current_bev_mask
from vis.reconstructed_udf_vis.render.checks.render_checks import check_render_state

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
    show_static: bool
    show_dynamic: bool
    show_trajectory: bool
    follow_ego: bool
    frame_index: int
    color_mode: str


def render_voxels(data, state, cfg, bev_cfg):
    """组合静态和当前帧动态体素；跟随模式仅保留自车 BEV 范围。"""
    check_render_state(data, state)
    static = _static_values(data, state, cfg, bev_cfg)
    dynamic = _dynamic_values(data, state, cfg, bev_cfg)
    values = [torch.cat((left, right)) for left, right in zip(static, dynamic)]
    points, udf, weight, tags, sources, actors = values
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.numpy().astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(
        _colors(udf, weight, tags, sources, actors, state.color_mode, cfg).numpy())
    return cloud


def _static_values(data, state, cfg, bev_cfg):
    field = data.static
    if not state.show_static:
        return _empty_values()
    points = (field["voxel_coords"].to(torch.float32) + 0.5) * field["voxel_size_m"]
    indices = torch.arange(len(points))
    if state.follow_ego:
        keep = torch.from_numpy(current_bev_mask(
            points.numpy(), data.ego_pose[state.frame_index].numpy(), bev_cfg))
        indices = indices[keep]
    indices = _uniform(indices, cfg.max_static_voxels)
    count = len(indices)
    return (points[indices], field["udf"][indices] / field["truncation_m"],
            field["weight"][indices], field["obj_tag"][indices],
            torch.zeros(count, dtype=torch.uint8),
            torch.full((count,), -1, dtype=torch.int64))


def _dynamic_values(data, state, cfg, bev_cfg):
    if not state.show_dynamic:
        return _empty_values()
    pose_indices = torch.nonzero(
        data.poses["frame_index"] == state.frame_index, as_tuple=False).flatten()
    objects = data.poses["object_index"][pose_indices]
    offsets = data.dynamic["voxel_offsets"]
    visible = offsets[objects + 1] > offsets[objects]
    pose_indices, objects = pose_indices[visible], objects[visible]
    if not len(objects):
        return _empty_values()
    chunks = [[] for _ in range(5)]
    counts = []
    for object_index in objects.tolist():
        first, last = int(offsets[object_index]), int(offsets[object_index + 1])
        count = last - first
        counts.append(count)
        chunks[0].append((data.dynamic["voxel_coords_local"][first:last].to(torch.float32)
                          + 0.5) * data.dynamic["voxel_size_m"])
        chunks[1].append(data.dynamic["udf"][first:last] / data.dynamic["truncation_m"])
        chunks[2].append(data.dynamic["weight"][first:last])
        chunks[3].append(data.dynamic["obj_tag"][first:last])
        chunks[4].append(data.dynamic["actor_id"][object_index].repeat(count))
    points, udf, weight, tags, actors = (torch.cat(chunk) for chunk in chunks)
    vertex_pose = torch.repeat_interleave(torch.arange(len(objects)), torch.tensor(counts))
    matrices = _pose_matrices(data.poses["transform"][pose_indices])
    points = torch.einsum("nij,nj->ni", matrices[vertex_pose, :3, :3], points) \
        + matrices[vertex_pose, :3, 3]
    indices = torch.arange(len(points))
    if state.follow_ego:
        keep = torch.from_numpy(current_bev_mask(
            points.numpy(), data.ego_pose[state.frame_index].numpy(), bev_cfg))
        indices = indices[keep]
    indices = _uniform(indices, cfg.max_dynamic_voxels)
    return (points[indices], udf[indices], weight[indices], tags[indices],
            torch.ones(len(indices), dtype=torch.uint8), actors[indices])


def _colors(udf, weight, tags, sources, actors, mode, cfg):
    if not len(udf):
        return torch.empty((0, 3), dtype=torch.float64)
    if mode == "udf":
        value = udf.to(torch.float64).clamp(0, 1)
        return torch.stack((value, 1 - torch.abs(value * 2 - 1), 1 - value), dim=1)
    if mode == "weight":
        value = weight.to(torch.float64).clamp(0, 1)
        return value[:, None].expand(-1, 3)
    if mode == "semantic":
        safe = tags.to(torch.int64).clamp(0, len(_SEMANTIC_RGB) - 1)
        return _SEMANTIC_RGB[safe]
    if mode == "source":
        static = torch.tensor(cfg.static_rgb, dtype=torch.float64) / 255
        dynamic = torch.tensor(cfg.dynamic_rgb, dtype=torch.float64) / 255
        return torch.where(sources[:, None].bool(), dynamic, static)
    value = actors.to(torch.float64).clamp_min(0)
    return torch.stack((
        torch.frac(value * 0.61803398875), torch.frac(value * 0.38196601125 + 0.3),
        torch.frac(value * 0.2360679775 + 0.6)), dim=1)


def _uniform(indices, limit):
    if len(indices) <= limit:
        return indices
    positions = torch.linspace(0, len(indices) - 1, limit).to(torch.int64)
    return indices[positions]


def _empty_values():
    return (torch.empty((0, 3), dtype=torch.float32),
            torch.empty(0, dtype=torch.float32), torch.empty(0, dtype=torch.float32),
            torch.empty(0, dtype=torch.uint8), torch.empty(0, dtype=torch.uint8),
            torch.empty(0, dtype=torch.int64))


def _pose_matrices(poses):
    radians = torch.deg2rad(poses[:, 3:6])
    cr, cp, cy = torch.cos(radians).T
    sr, sp, sy = torch.sin(radians).T
    matrices = torch.eye(4).repeat(len(poses), 1, 1)
    matrices[:, :3, 3] = poses[:, :3]
    matrices[:, 0, :3] = torch.stack((
        cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr), dim=1)
    matrices[:, 1, :3] = torch.stack((
        sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr), dim=1)
    matrices[:, 2, :3] = torch.stack((sp, -cp * sr, cp * cr), dim=1)
    return matrices


__all__ = ["RenderState", "render_voxels", "render_trajectories"]
