"""LiDAR 点云体素化：在 CPU 上向量化计算每格 XYZ 均值与总体标准差。

模块: data/lidar_voxelization/lidar_voxelization.py
依赖: torch, data.lidar_voxelization.checks.lidar_voxelization_checks
读取配置: —（BEV XYZ 量程与 voxel_size_m 由调用方传入，来源 config.model.driving）
对外接口:
    - lidar_xyz_to_voxels(points_xyz, lidar_extrinsic, ego_box, bev, voxel_size_m) -> tuple[Tensor, Tensor]
说明: 原始点先平移到 ego 系并剔除自车有向 Box 内点，再在半开区间内体素化；输出 X 行反转，使首行为 BEV 远端。
      六通道依次为 XYZ 均值与 XYZ 总体标准差，单点体素标准差严格为零。
"""

from __future__ import annotations

import torch

from data.lidar_voxelization.checks.lidar_voxelization_checks import (
    check_lidar_voxel_inputs,
)


__all__ = ["lidar_xyz_to_voxels"]


def lidar_xyz_to_voxels(points_xyz, lidar_extrinsic, ego_box, bev, voxel_size_m):
    """把 `[N,3]` LiDAR 点编码为 `[6,Z,X,Y]` 统计与 `[1,Z,X,Y]` 占用掩码。"""
    points = torch.as_tensor(points_xyz, dtype=torch.float32, device="cpu")
    box_transform = torch.as_tensor(
        ego_box["transform"], dtype=torch.float32, device="cpu")
    box_extent = torch.as_tensor(
        ego_box["extent"], dtype=torch.float32, device="cpu")
    check_lidar_voxel_inputs(
        points, lidar_extrinsic, box_transform, box_extent, voxel_size_m)
    lower = torch.tensor(
        [bev.x_min_m, bev.y_min_m, bev.z_min_m], dtype=torch.float32)
    upper = torch.tensor(
        [bev.x_max_m, bev.y_max_m, bev.z_max_m], dtype=torch.float32)
    counts_xyz = torch.round((upper - lower) / voxel_size_m).to(torch.long)
    nx, ny, nz = (int(value) for value in counts_xyz)
    voxel_count = nx * ny * nz
    if points.numel() == 0:
        return (
            torch.zeros((6, nz, nx, ny), dtype=torch.float32),
            torch.zeros((1, nz, nx, ny), dtype=torch.bool),
        )

    points = points + torch.as_tensor(lidar_extrinsic, dtype=torch.float32)
    box_local = (
        points - box_transform[:3, 3]
    ) @ box_transform[:3, :3]
    points = points[torch.any(box_local.abs() > box_extent, dim=1)]
    valid = torch.all((points >= lower) & (points < upper), dim=1)
    points = points[valid]
    if points.numel() == 0:
        return (
            torch.zeros((6, nz, nx, ny), dtype=torch.float32),
            torch.zeros((1, nz, nx, ny), dtype=torch.bool),
        )

    indices = torch.floor((points - lower) / voxel_size_m).to(torch.long)
    flat = (indices[:, 0] * ny + indices[:, 1]) * nz + indices[:, 2]
    occupied_count = torch.bincount(flat, minlength=voxel_count)
    scatter_index = flat[None].expand(3, -1)
    sums = torch.zeros((3, voxel_count), dtype=torch.float32).scatter_add_(
        1, scatter_index, points.transpose(0, 1))
    denominator = occupied_count.clamp_min(1).to(torch.float32)
    means = sums / denominator
    centered = points.transpose(0, 1) - means.gather(1, scatter_index)
    centered_square_sums = torch.zeros_like(sums).scatter_add_(
        1, scatter_index, centered.square())
    variances = (centered_square_sums / denominator).clamp_min(0.0)
    standard_deviations = torch.sqrt(variances)

    stats = torch.cat((means, standard_deviations), dim=0)
    stats = stats.reshape(6, nx, ny, nz).permute(0, 3, 1, 2).flip(2).contiguous()
    occupied = (occupied_count > 0).reshape(nx, ny, nz).permute(2, 0, 1).flip(1)
    return stats, occupied[None].contiguous()
