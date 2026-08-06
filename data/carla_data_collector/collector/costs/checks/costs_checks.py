import numpy as np


def check_cost_inputs(world_states, model_steps, route_geometry):
    """校验对象: evaluate_drive_costs 入参——时间轴、轨迹和路线几何必须可对齐。"""
    if not world_states or not model_steps:
        raise ValueError("GT世界状态与模型步骤均不得为空")
    frame_ids = [int(state["frame_id"]) for state in world_states]
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError("GT世界状态 frame_id 不得重复")
    points = np.asarray(route_geometry.get("points"), dtype=np.float64)
    arc = np.asarray(route_geometry.get("arc_m"), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2 or len(points) < 2 or arc.shape != (len(points),):
        raise ValueError("route_geometry 期望 points[N,>=2] 与 arc_m[N]")
    for step in model_steps:
        trajectories = np.asarray(step["trajectories"])
        if trajectories.ndim != 3 or trajectories.shape[-1] != 2 \
                or not np.all(np.isfinite(trajectories)):
            raise ValueError("模型候选轨迹期望有限 [M,T,2]")
