# 本文件为 model/driving_model/driving_model.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_driving_inputs(rgb, intrinsics, extrinsics, target_point, ego_velocity, previous_rgb,
                         previous_to_current, previous_valid):
    """校验对象: DrivingModel.forward 入参 —— 图像、标定、规划条件与帧间变换的形状批次须一致。"""
    if rgb.ndim != 4 or int(rgb.shape[1]) != 3:
        raise ValueError("rgb 期望 (B,3,H,W)，实际 {}。".format(tuple(rgb.shape)))
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
        if tensor.ndim != 2 or int(tensor.shape[1]) != dim or int(tensor.shape[0]) != b:
            raise ValueError("{} 期望 ({}, {})，实际 {}。".format(name, b, dim, tuple(tensor.shape)))
    if previous_to_current.ndim != 3 or tuple(previous_to_current.shape[1:]) != (3, 3) \
            or int(previous_to_current.shape[0]) != b:
        raise ValueError("previous_to_current 期望 ({},3,3)，实际 {}。".format(
            b, tuple(previous_to_current.shape)))
    if previous_valid.ndim != 1 or int(previous_valid.shape[0]) != b:
        raise ValueError("previous_valid 期望 ({},)，实际 {}。".format(b, tuple(previous_valid.shape)))
