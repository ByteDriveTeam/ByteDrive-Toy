"""UnifiedDrivingModel 的输入检查。"""

import torch


def check_unified_driving_inputs(rgb, intrinsics, navigation_command, ego_state,
                                 motion_matrices, num_commands, recovery_command=None):
    """检查对象: UnifiedDrivingModel.forward 的四帧 RGB、相机、条件和运动矩阵。"""
    if rgb.ndim != 5 or tuple(rgb.shape[1:3]) != (4, 3):
        raise ValueError("新 DrivingModel 的 rgb 必须为 [B,4,3,H,W]")
    batch = int(rgb.shape[0])
    if recovery_command is not None:
        if (recovery_command.shape != (batch,) or recovery_command.dtype != torch.long
                or recovery_command.device != navigation_command.device):
            raise ValueError("recovery_command must be int64 [B] on navigation device")
        if recovery_command.device.type == "cpu" and bool(
                ((recovery_command < 0) | (recovery_command > 2)).any()):
            raise ValueError("recovery_command must be 0/1/2")
    if intrinsics.ndim != 3 or tuple(intrinsics.shape) != (batch, 4, 4):
        raise ValueError("intrinsics 必须为 [B,4,4]，每行是 fx/fy/cx/cy")
    if navigation_command.ndim != 1 or navigation_command.shape[0] != batch \
            or navigation_command.dtype != torch.long:
        raise ValueError("navigation_command 必须为 [B] 的有效整数类别 ID")
    if navigation_command.device.type == "cpu" \
            and bool(((navigation_command < 0) | (navigation_command >= num_commands)).any()):
        raise ValueError("navigation_command 类别 ID 超出配置的导航类别范围")
    if ego_state.ndim != 2 or tuple(ego_state.shape) != (batch, 4):
        raise ValueError("ego_state 必须为 [B,4]")
    if motion_matrices.ndim != 4 or tuple(motion_matrices.shape[:2]) != (batch, 4) \
            or tuple(motion_matrices.shape[-2:]) not in ((3, 3), (4, 4)):
        raise ValueError("motion_matrices 必须为 [B,4,3,3] 或 [B,4,4,4]")
