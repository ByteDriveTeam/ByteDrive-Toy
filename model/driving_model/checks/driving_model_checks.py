"""驾驶模型五帧输入校验。"""


def check_driving_inputs(rgb, intrinsics, extrinsics, target_point, ego_velocity,
                         history_rgb, history_to_current, history_valid):
    """校验对象: DrivingModel.forward —— 五帧三目与四维刚性变换须按批次对齐。"""
    batch = rgb.shape[0]
    if rgb.ndim != 5 or tuple(rgb.shape[1:3]) != (3, 3):
        raise ValueError("rgb 须 [B,3,3,H,W]")
    if tuple(history_rgb.shape) != (batch, 4, *rgb.shape[1:]):
        raise ValueError("history_rgb 须 [B,4,3,3,H,W]")
    if tuple(history_to_current.shape) != (batch, 4, 4, 4):
        raise ValueError("history_to_current 须 [B,4,4,4]")
    if tuple(history_valid.shape) != (batch, 4):
        raise ValueError("history_valid 须 [B,4]")
    if tuple(intrinsics.shape) != (batch, 3, 4) or tuple(extrinsics.shape) != (batch, 3, 6):
        raise ValueError("三目内外参形状非法")
    if tuple(target_point.shape) != (batch, 2) or tuple(ego_velocity.shape) != (batch, 2):
        raise ValueError("规划条件须为 [B,2]")
