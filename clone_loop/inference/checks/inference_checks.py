import numpy as np
import torch


def check_frame(frame, views, height, width):
    """校验对象: ClosedLoopPolicy.infer 的 frame_bgr —— 必须匹配三目配置。"""
    if frame.shape != (views, height, width, 3) or frame.dtype != np.uint8:
        raise ValueError("闭环 RGB 期望 ({},{},{},3) uint8".format(
            views, height, width))


def check_lidar(points):
    """校验对象: ClosedLoopPolicy.infer 的 lidar_xyz —— 必须为有限 FP32 `[N,3]`。"""
    if points.ndim != 2 or points.shape[1:] != (3,) or points.dtype != np.float32:
        raise ValueError("闭环 LiDAR 期望 [N,3] float32")
    if not np.all(np.isfinite(points)):
        raise ValueError("闭环 LiDAR 包含非有限数")


def check_observation(observation):
    """校验对象: ClosedLoopPolicy.infer 的 observation —— 模型条件字段必须齐全且有限。"""
    expected = {
        "pose", "intrinsics", "extrinsics", "target_point", "ego_velocity",
        "lidar_count", "lidar_valid", "ego_box",
    }
    missing = expected.difference(observation)
    if missing:
        raise ValueError("闭环观测缺少字段: {}".format(sorted(missing)))
    numeric = expected.difference({"lidar_valid", "ego_box"})
    if not all(np.all(np.isfinite(observation[key])) for key in numeric):
        raise ValueError("闭环观测包含非有限数")
    if np.shape(observation["intrinsics"]) != (3, 4) \
            or np.shape(observation["extrinsics"]) != (3, 6):
        raise ValueError("闭环三目标定期望 intrinsics [3,4]、extrinsics [3,6]")
    ego_box = observation["ego_box"]
    if not isinstance(ego_box, dict) \
            or any(np.shape(ego_box.get(key)) != (3,)
                   for key in ("location", "extent", "rotation")):
        raise ValueError("闭环 ego_box 期望三维 location/extent/rotation")
    if not all(np.all(np.isfinite(ego_box[key]))
               for key in ("location", "extent", "rotation")) \
            or not np.all(np.asarray(ego_box["extent"]) > 0):
        raise ValueError("闭环 ego_box 必须有限且 extent 为正")


def check_trajectory_candidates(trajectories, max_abs_waypoint_m):
    """校验对象: 模型 trajectories —— 在线控制前必须全部有限且未明显发散。"""
    if trajectories.ndim != 3 or trajectories.shape[-1] != 2:
        raise ValueError("模型 trajectories 期望 [M,T,2]")
    if not bool(torch.isfinite(trajectories).all()):
        raise RuntimeError("模型轨迹包含 NaN/Inf，闭环已拒绝执行")
    if float(trajectories.abs().max()) > max_abs_waypoint_m:
        raise RuntimeError("模型轨迹超出闭环允许的绝对坐标范围")
