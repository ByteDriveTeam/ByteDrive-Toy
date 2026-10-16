"""UnifiedDrivingModel 的输入检查。"""


def check_unified_driving_inputs(rgb, intrinsics, target_point, ego_state, motion_matrices):
    """检查对象: UnifiedDrivingModel.forward 的四帧 RGB、相机、条件和运动矩阵。"""
    if rgb.ndim != 5 or tuple(rgb.shape[1:3]) != (4, 3):
        raise ValueError("新 DrivingModel 的 rgb 必须为 [B,4,3,H,W]")
    batch = int(rgb.shape[0])
    if intrinsics.ndim != 3 or tuple(intrinsics.shape) != (batch, 4, 4):
        raise ValueError("intrinsics 必须为 [B,4,4]，每行是 fx/fy/cx/cy")
    for name, tensor, width in (("target_point", target_point, 2), ("ego_state", ego_state, 4)):
        if tensor.ndim != 2 or tuple(tensor.shape) != (batch, width):
            raise ValueError("{} 必须为 [B,{}]".format(name, width))
    if motion_matrices.ndim != 4 or tuple(motion_matrices.shape[:2]) != (batch, 4) \
            or tuple(motion_matrices.shape[-2:]) not in ((3, 3), (4, 4)):
        raise ValueError("motion_matrices 必须为 [B,4,3,3] 或 [B,4,4,4]")
