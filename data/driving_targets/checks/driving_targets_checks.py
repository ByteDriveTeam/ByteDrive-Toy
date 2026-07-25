# 本文件为 data/driving_targets/driving_targets.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_bev_params(bev):
    """校验对象: BevParams —— 量程 min<max、分辨率为正。"""
    assert bev.x_min < bev.x_max and bev.y_min < bev.y_max, "BevParams 需 x_min<x_max 且 y_min<y_max"
    assert bev.height > 0 and bev.width > 0, "BevParams.height/width 必须 > 0"


def check_multicamera_inputs(depth_maps, intrinsics, extrinsics):
    """校验对象: 多相机监督几何入参 —— 深度 [V,Hc,Wc]、内外参与相机数一致。"""
    if depth_maps.ndim != 3:
        raise ValueError("depth_maps 期望三维 (V,Hc,Wc)，实际 {}。".format(tuple(depth_maps.shape)))
    views = int(depth_maps.shape[0])
    intrinsic_shape_ok = (
        len(intrinsics) == views
        and (all({"fx", "fy", "cx", "cy"}.issubset(item) for item in intrinsics)
             if views and isinstance(intrinsics[0], dict)
             else getattr(intrinsics, "shape", None) == (views, 4))
    )
    if not intrinsic_shape_ok or extrinsics.shape != (views, 6):
        raise ValueError("多相机深度/内参/外参相机数或形状不一致：{} / {} / {}。".format(
            tuple(depth_maps.shape), len(intrinsics), tuple(extrinsics.shape)))


def check_visible_moving_box_inputs(depth_maps, intrinsics, extrinsics, min_visible_pixels):
    """校验对象: visible_moving_box_occupancy 入参 —— 三目几何一致且可见阈值不少于 10 像素。"""
    check_multicamera_inputs(depth_maps, intrinsics, extrinsics)
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
    """校验对象: behavior_targets 入参 —— 航点 [K,2]、掩码 [K] 与三目 Seg。"""
    if waypoints.ndim != 2 or int(waypoints.shape[1]) != 2:
        raise ValueError("waypoints 期望 [K,2]，实际 {}。".format(tuple(waypoints.shape)))
    if valid.ndim != 1 or len(valid) != len(waypoints):
        raise ValueError("valid 期望 [K] 且与航点同长度，实际 {}。".format(tuple(valid.shape)))
    if semantic.ndim != 3:
        raise ValueError("semantic 期望三维 Seg (V,H,W)，实际 {}。".format(tuple(semantic.shape)))
