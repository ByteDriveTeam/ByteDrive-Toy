import torch


def check_lidar_voxel_inputs(
        points_xyz, lidar_extrinsic, ego_box_transform, ego_box_extent,
        voxel_size_m):
    """校验对象: lidar_xyz_to_voxels 入参 —— 点云、平移外参、自车 Box 与体素边长须合法。"""
    if points_xyz.ndim != 2 or int(points_xyz.shape[1]) != 3:
        raise ValueError("points_xyz 期望 [N,3]，实际 {}。".format(tuple(points_xyz.shape)))
    if len(lidar_extrinsic) != 3:
        raise ValueError("lidar_extrinsic 期望长度 3，实际 {}。".format(len(lidar_extrinsic)))
    if ego_box_transform.shape != (4, 4) or not bool(torch.isfinite(ego_box_transform).all()):
        raise ValueError("ego_box.transform 期望有限 [4,4] 矩阵。")
    if ego_box_extent.shape != (3,) or not bool(torch.isfinite(ego_box_extent).all()) \
            or not bool(torch.all(ego_box_extent > 0)):
        raise ValueError("ego_box.extent 期望有限正数 [3]。")
    if voxel_size_m <= 0:
        raise ValueError("voxel_size_m 必须 > 0，实际 {}。".format(voxel_size_m))
