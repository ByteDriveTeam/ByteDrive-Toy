# 本文件为 model/driving_model/driving_model.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_driving_inputs(rgb, intrinsics, extrinsics, target_point, ego_velocity, previous_rgb,
                         previous_to_current, previous_valid, lidar_stats=None,
                         lidar_occupied=None, lidar_valid=None, previous_lidar_stats=None,
                         previous_lidar_occupied=None, previous_lidar_valid=None):
    """校验对象: DrivingModel.forward 入参 —— 图像、规划条件、时序变换及可选 LiDAR 须一致。"""
    if rgb.ndim != 5 or int(rgb.shape[1]) != 3 or int(rgb.shape[2]) != 3:
        raise ValueError("rgb 期望 (B,3,3,H,W)，实际 {}。".format(tuple(rgb.shape)))
    b = int(rgb.shape[0])
    if previous_rgb.shape != rgb.shape:
        raise ValueError("previous_rgb 须与 rgb 同形，实际 {} / {}。".format(
            tuple(previous_rgb.shape), tuple(rgb.shape)))
    if target_point.ndim != 2 or tuple(target_point.shape) != (b, 2):
        raise ValueError("target_point 期望 ({},2)，实际 {}。".format(b, tuple(target_point.shape)))
    if ego_velocity.shape != target_point.shape:
        raise ValueError("ego_velocity 须与 target_point 同为 [B,2]，实际 {} / {}。".format(
            tuple(ego_velocity.shape), tuple(target_point.shape)))
    for name, tensor, dim in (("intrinsics", intrinsics, 4), ("extrinsics", extrinsics, 6)):
        if tensor.ndim != 3 or tuple(tensor.shape) != (b, 3, dim):
            raise ValueError("{} 期望 ({},3,{}), 实际 {}。".format(
                name, b, dim, tuple(tensor.shape)))
    if previous_to_current.ndim != 3 or tuple(previous_to_current.shape[1:]) != (3, 3) \
            or int(previous_to_current.shape[0]) != b:
        raise ValueError("previous_to_current 期望 ({},3,3)，实际 {}。".format(
            b, tuple(previous_to_current.shape)))
    if previous_valid.ndim != 1 or int(previous_valid.shape[0]) != b:
        raise ValueError("previous_valid 期望 ({},)，实际 {}。".format(b, tuple(previous_valid.shape)))
    current_lidar = (lidar_stats, lidar_occupied, lidar_valid)
    previous_lidar = (
        previous_lidar_stats, previous_lidar_occupied, previous_lidar_valid)
    for name, values in (("当前", current_lidar), ("历史", previous_lidar)):
        present = tuple(value is not None for value in values)
        if any(present) and not all(present):
            raise ValueError("{} LiDAR 的 stats/occupied/valid 必须同时提供。".format(name))
