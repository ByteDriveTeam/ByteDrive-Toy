def check_lidar_voxel_inputs(points_xyz, lidar_extrinsic, voxel_size_m):
    """校验对象: lidar_xyz_to_voxels 入参 —— 点云、平移外参与体素边长须合法。"""
    if points_xyz.ndim != 2 or int(points_xyz.shape[1]) != 3:
        raise ValueError("points_xyz 期望 [N,3]，实际 {}。".format(tuple(points_xyz.shape)))
    if len(lidar_extrinsic) != 3:
        raise ValueError("lidar_extrinsic 期望长度 3，实际 {}。".format(len(lidar_extrinsic)))
    if voxel_size_m <= 0:
        raise ValueError("voxel_size_m 必须 > 0，实际 {}。".format(voxel_size_m))
