import numpy as np


def check_control_inputs(trajectory, pose, sim_time_s, speed_mps,
                         behavior_probabilities, stop_indices):
    """校验对象: TrajectoryController.command 入参 —— 轨迹、位姿与状态必须有限且形状正确。"""
    if trajectory.ndim != 2 or trajectory.shape[1] != 2 or len(trajectory) == 0:
        raise ValueError("trajectory 期望 [T,2] 且 T>0")
    if pose.shape != (6,):
        raise ValueError("pose 期望 CARLA [x,y,z,roll,pitch,yaw] 六维位姿")
    if not np.all(np.isfinite(trajectory)) or not np.all(np.isfinite(pose)) \
            or not np.isfinite(sim_time_s) or not np.isfinite(speed_mps):
        raise ValueError("轨迹、位姿、仿真时间和速度必须为有限数")
    if float(speed_mps) < 0:
        raise ValueError("speed_mps 必须为非负数")
    if behavior_probabilities.ndim != 1 or len(behavior_probabilities) == 0 \
            or not np.all(np.isfinite(behavior_probabilities)) \
            or np.any(behavior_probabilities < 0) \
            or np.any(behavior_probabilities > 1):
        raise ValueError("behavior_probabilities 期望一维且全部位于 [0,1]")
    if max(stop_indices) >= len(behavior_probabilities):
        raise ValueError("behavior_probabilities 未覆盖配置的停车标签索引")
