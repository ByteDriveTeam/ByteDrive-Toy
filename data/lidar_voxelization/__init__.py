"""LiDAR 点云体素均值与标准差编码的公开 API 重导出入口。

模块: data/lidar_voxelization/__init__.py
依赖: data.lidar_voxelization.lidar_voxelization
读取配置: —（量程与体素尺寸由调用方传入，来源 config.model.driving）
对外接口:
    - lidar_xyz_to_voxels(points_xyz, lidar_extrinsic, ego_box, bev, voxel_size_m) -> tuple[Tensor, Tensor]
"""

from data.lidar_voxelization.lidar_voxelization import lidar_xyz_to_voxels

__all__ = ["lidar_xyz_to_voxels"]
