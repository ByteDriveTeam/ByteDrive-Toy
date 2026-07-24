import numpy as np
import torch


def check_frame(frame, height, width):
    """校验对象: ClosedLoopPolicy.infer 的 frame_bgr —— 必须匹配配置相机。"""
    if frame.shape != (height, width, 3) or frame.dtype != np.uint8:
        raise ValueError("闭环 RGB 期望 ({},{},3) uint8".format(height, width))


def check_observation(observation):
    """校验对象: ClosedLoopPolicy.infer 的 observation —— 模型条件字段必须齐全且有限。"""
    expected = {"pose", "intrinsics", "extrinsics", "target_point", "ego_velocity"}
    missing = expected.difference(observation)
    if missing:
        raise ValueError("闭环观测缺少字段: {}".format(sorted(missing)))
    if not all(np.all(np.isfinite(observation[key])) for key in expected):
        raise ValueError("闭环观测包含非有限数")


def check_trajectory_candidates(trajectories, max_abs_waypoint_m):
    """校验对象: 模型 trajectories —— 在线控制前必须全部有限且未明显发散。"""
    if trajectories.ndim != 3 or trajectories.shape[-1] != 2:
        raise ValueError("模型 trajectories 期望 [M,T,2]")
    if not bool(torch.isfinite(trajectories).all()):
        raise RuntimeError("模型轨迹包含 NaN/Inf，闭环已拒绝执行")
    if float(trajectories.abs().max()) > max_abs_waypoint_m:
        raise RuntimeError("模型轨迹超出闭环允许的绝对坐标范围")
