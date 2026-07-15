# 本文件为 data/driving_targets/driving_targets.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_bev_params(bev):
    """校验对象: BevParams —— 量程 min<max、分辨率为正。"""
    assert bev.x_min < bev.x_max and bev.y_min < bev.y_max, "BevParams 需 x_min<x_max 且 y_min<y_max"
    assert bev.height > 0 and bev.width > 0, "BevParams.height/width 必须 > 0"


def check_depth_map(depth_m):
    """校验对象: risk_field/visible_moving_box_occupancy 入参 depth_m —— 二维深度图 (Hc, Wc)。"""
    if depth_m.ndim != 2:
        raise ValueError("depth_m 期望二维 (Hc,Wc)，实际 {}。".format(tuple(depth_m.shape)))


def check_visible_moving_box_inputs(depth_m, min_visible_pixels):
    """校验对象: visible_moving_box_occupancy 入参 —— 深度图为二维且可见阈值不少于 10 像素。"""
    check_depth_map(depth_m)
    if min_visible_pixels < 10:
        raise ValueError("min_visible_pixels 必须 >= 10，实际 {}。".format(min_visible_pixels))


def check_motion_sequence(world_velocities, sim_times):
    """校验对象: speed_accelerations 入参 —— 速度 [F,3] 与时间 [F] 帧数一致。"""
    if world_velocities.ndim != 2 or int(world_velocities.shape[1]) != 3:
        raise ValueError("world_velocities 期望 [F,3]，实际 {}。".format(tuple(world_velocities.shape)))
    if sim_times.ndim != 1 or len(sim_times) != len(world_velocities):
        raise ValueError("sim_times 期望 [F] 且与速度同帧数，实际 {} / {}。".format(
            tuple(sim_times.shape), len(world_velocities)))


def check_behavior_inputs(waypoints, valid, semantic):
    """校验对象: behavior_targets 入参 —— 航点 [K,2]、掩码 [K] 与二维 Seg。"""
    if waypoints.ndim != 2 or int(waypoints.shape[1]) != 2:
        raise ValueError("waypoints 期望 [K,2]，实际 {}。".format(tuple(waypoints.shape)))
    if valid.ndim != 1 or len(valid) != len(waypoints):
        raise ValueError("valid 期望 [K] 且与航点同长度，实际 {}。".format(tuple(valid.shape)))
    if semantic.ndim != 2:
        raise ValueError("semantic 期望二维 Seg (H,W)，实际 {}。".format(tuple(semantic.shape)))
