import numpy as np


def check_control_inputs(trajectory, speed_mps, behavior_probabilities, stop_indices):
    """校验对象: TrajectoryController.command 入参 —— 轨迹与状态必须有限且形状正确。"""
    if trajectory.ndim != 2 or trajectory.shape[1] != 2 or len(trajectory) == 0:
        raise ValueError("trajectory 期望 [T,2] 且 T>0")
    if not np.all(np.isfinite(trajectory)) or not np.isfinite(speed_mps):
        raise ValueError("轨迹和速度必须为有限数")
    if behavior_probabilities.ndim != 1 or not np.all(np.isfinite(behavior_probabilities)):
        raise ValueError("behavior_probabilities 期望有限一维数组")
    if max(stop_indices) >= len(behavior_probabilities):
        raise ValueError("停车行为索引超出模型输出范围")
