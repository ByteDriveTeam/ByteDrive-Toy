# 本文件为 model/driving_model/driving_model.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_driving_inputs(rgb, intrinsics, extrinsics, ego_velocity, target_point):
    """校验对象: DrivingModel.forward 入参 —— 各输入维度与批次一致（rgb 四维，其余 [B,·] 二维）。"""
    if rgb.ndim != 4 or int(rgb.shape[1]) != 3:
        raise ValueError("rgb 期望 (B,3,H,W)，实际 {}。".format(tuple(rgb.shape)))
    b = int(rgb.shape[0])
    for name, tensor, dim in (("intrinsics", intrinsics, 4), ("extrinsics", extrinsics, 6),
                              ("ego_velocity", ego_velocity, 2), ("target_point", target_point, 2)):
        if tensor.ndim != 2 or int(tensor.shape[1]) != dim or int(tensor.shape[0]) != b:
            raise ValueError("{} 期望 ({}, {})，实际 {}。".format(name, b, dim, tuple(tensor.shape)))
