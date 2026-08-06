"""使用未来10Hz GT Box轨迹离线标注候选、当前位置、下一刻与历史原始代价。

模块: collector/costs/costs.py
依赖: math, numpy, collector.costs.checks.costs_checks
读取配置:
    carla_collector.cost.safety.safe_clearance_m
    carla_collector.cost.compliance.route_margin_m / wrong_way_tolerance_deg / speed_tolerance_mps
    carla_collector.cost.interaction.min_gap_m / reaction_time_s / lateral_margin_m
    carla_collector.cost.efficiency.max_reference_speed_mps
    carla_collector.cost.comfort_control.stationary_speed_mps
    carla_collector.model_collection.stop_obstacle_distance_m / stop_obstacle_half_angle_deg /
        stop_red_light_distance_m
对外接口:
    - COST_TERMS
    - evaluate_drive_costs(world_states, model_steps, route_geometry, cfg_cost, cfg_model) -> None
说明: 所有项均为非负原始物理量或事件计数，无上限、无归一化、无跨项加权。候选环境采用
      Winner真实执行后采得的未来GT actor Box；灯态固定为候选生成时刻的真值。
"""

import math

import numpy as np

from collector.costs.checks.costs_checks import check_cost_inputs


COST_TERMS = (
    {"name": "safety.clearance_deficit_m", "category": "safety", "unit": "m"},
    {"name": "safety.overlap_depth_m", "category": "safety", "unit": "m"},
    {"name": "safety.collision_events", "category": "safety", "unit": "count"},
    {"name": "compliance.route_overflow_m", "category": "compliance", "unit": "m"},
    {"name": "compliance.wrong_way_excess_deg", "category": "compliance", "unit": "deg"},
    {"name": "compliance.overspeed_excess_mps", "category": "compliance", "unit": "m/s"},
    {"name": "compliance.red_light_crossings", "category": "compliance", "unit": "count"},
    {"name": "interaction.courtesy_intrusion_m", "category": "interaction", "unit": "m"},
    {"name": "interaction.required_deceleration_mps2", "category": "interaction", "unit": "m/s2"},
    {"name": "efficiency.progress_shortfall_mps", "category": "efficiency", "unit": "m/s"},
    {"name": "efficiency.reverse_progress_mps", "category": "efficiency", "unit": "m/s"},
    {"name": "efficiency.unjustified_stationary_s", "category": "efficiency", "unit": "s"},
    {"name": "comfort_control.longitudinal_acceleration_mps2", "category": "comfort_control", "unit": "m/s2"},
    {"name": "comfort_control.lateral_acceleration_mps2", "category": "comfort_control", "unit": "m/s2"},
    {"name": "comfort_control.jerk_mps3", "category": "comfort_control", "unit": "m/s3"},
    {"name": "comfort_control.yaw_rate_radps", "category": "comfort_control", "unit": "rad/s"},
    {"name": "comfort_control.steer_delta", "category": "comfort_control", "unit": "ratio"},
    {"name": "comfort_control.throttle_delta", "category": "comfort_control", "unit": "ratio"},
    {"name": "comfort_control.brake_delta", "category": "comfort_control", "unit": "ratio"},
    {"name": "comfort_control.throttle_brake_overlap", "category": "comfort_control", "unit": "ratio2"},
)

_TERM_INDEX = {item["name"]: index for index, item in enumerate(COST_TERMS)}
_ACTUAL_ONLY = tuple(_TERM_INDEX[name] for name in (
    "safety.collision_events", "comfort_control.steer_delta",
    "comfort_control.throttle_delta", "comfort_control.brake_delta",
    "comfort_control.throttle_brake_overlap"))


def _rotation(yaw_deg):
    yaw = math.radians(float(yaw_deg))
    return np.array([[math.cos(yaw), -math.sin(yaw)],
                     [math.sin(yaw), math.cos(yaw)]], dtype=np.float64)


def _corners(box):
    center = np.asarray(box["location"][:2], dtype=np.float64)
    extent = np.asarray(box["extent"][:2], dtype=np.float64)
    local = np.array([[extent[0], extent[1]], [extent[0], -extent[1]],
                      [-extent[0], -extent[1]], [-extent[0], extent[1]]])
    return local @ _rotation(box["rotation"][2]).T + center


def _axes(corners):
    edges = np.roll(corners, -1, axis=0) - corners
    normals = np.stack((-edges[:, 1], edges[:, 0]), axis=1)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(lengths, np.finfo(np.float64).eps)


def _point_segment_distance(point, start, end):
    delta = end - start
    scale = float(np.dot(delta, delta))
    ratio = 0.0 if scale <= np.finfo(np.float64).eps else float(
        np.clip(np.dot(point - start, delta) / scale, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + ratio * delta)))


def _signed_box_distance(first, second):
    """返回二维OBB有符号净距；重叠时为负的最小穿透深度。"""
    a, b = _corners(first), _corners(second)
    overlaps = []
    separated = False
    for axis in np.vstack((_axes(a), _axes(b))):
        pa, pb = a @ axis, b @ axis
        overlap = min(pa.max(), pb.max()) - max(pa.min(), pb.min())
        overlaps.append(float(overlap))
        separated = separated or overlap < 0.0
    if not separated:
        return -min(overlaps)
    distances = [
        _point_segment_distance(point, edge[0], edge[1])
        for source, target in ((a, b), (b, a))
        for point in source
        for edge in zip(target, np.roll(target, -1, axis=0))
    ]
    return min(distances)


def _route_projection(point, route):
    points = route["points"]
    index = int(np.argmin(np.linalg.norm(points - point[None], axis=1)))
    return index, float(route["arc"][index])


def _angle_difference(first_deg, second_deg):
    return abs((float(first_deg) - float(second_deg) + 180.0) % 360.0 - 180.0)


def _segments_intersect(a, b, c, d):
    def cross(u, v):
        return float(u[0] * v[1] - u[1] * v[0])
    ab, cd = b - a, d - c
    denominator = cross(ab, cd)
    if abs(denominator) <= np.finfo(np.float64).eps:
        return False
    t = cross(c - a, cd) / denominator
    u = cross(c - a, ab) / denominator
    return 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0


def _red_crossing(previous_xy, current_xy, relevant):
    if not relevant.get("valid") or relevant.get("state") != "red":
        return 0.0
    center = np.asarray(relevant["stop_location"][:2], dtype=np.float64)
    yaw = math.radians(float(relevant["stop_yaw"]))
    lateral = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)
    half = float(relevant["lane_width"]) * 0.5
    return float(_segments_intersect(previous_xy, current_xy,
                                     center - lateral * half, center + lateral * half))


def _ego_box_at(state, local_xy, local_yaw_deg):
    ego = state["ego"]
    actual = next(box for box in state["bboxes"] if box["semantic"] == "ego")
    origin = np.asarray(ego["transform"][:2], dtype=np.float64)
    base_yaw = float(ego["transform"][5])
    world_xy = origin + _rotation(base_yaw) @ np.asarray(local_xy, dtype=np.float64)
    offset = _rotation(-base_yaw) @ (
        np.asarray(actual["location"][:2], dtype=np.float64) - origin)
    yaw = base_yaw + float(local_yaw_deg)
    center = world_xy + _rotation(yaw) @ offset
    return {"location": [center[0], center[1], actual["location"][2]],
            "extent": list(actual["extent"]), "rotation": [0.0, 0.0, yaw]}


def _motion_box(state, previous_xy, current_xy, speed, yaw_deg, collision_events,
                previous_control, cfg_cost, cfg_model, route):
    box = state["candidate_box"]
    actors = [item for item in state["bboxes"] if item["semantic"] != "ego"]
    distances = [_signed_box_distance(box, actor) for actor in actors]
    minimum = min(distances) if distances else float("inf")
    values = np.zeros(len(COST_TERMS), dtype=np.float64)
    values[_TERM_INDEX["safety.clearance_deficit_m"]] = max(
        0.0, cfg_cost.safety.safe_clearance_m - minimum)
    values[_TERM_INDEX["safety.overlap_depth_m"]] = max(0.0, -minimum)
    values[_TERM_INDEX["safety.collision_events"]] = float(collision_events)

    corners = _corners(box)
    route_indices = [int(np.argmin(np.linalg.norm(
        route["points"] - corner[None], axis=1))) for corner in corners]
    overflow = [max(0.0, float(np.linalg.norm(corner - route["points"][idx]))
                    - route["lane_width"][idx] * 0.5 - cfg_cost.compliance.route_margin_m)
                for corner, idx in zip(corners, route_indices)]
    route_index, route_arc = _route_projection(np.asarray(box["location"][:2]), route)
    values[_TERM_INDEX["compliance.route_overflow_m"]] = max(overflow)
    values[_TERM_INDEX["compliance.wrong_way_excess_deg"]] = max(
        0.0, _angle_difference(yaw_deg, route["yaw"][route_index])
        - cfg_cost.compliance.wrong_way_tolerance_deg)
    values[_TERM_INDEX["compliance.overspeed_excess_mps"]] = max(
        0.0, speed - state["speed_limit_mps"] - cfg_cost.compliance.speed_tolerance_mps)
    values[_TERM_INDEX["compliance.red_light_crossings"]] = _red_crossing(
        previous_xy, current_xy, state["relevant_traffic_control"])

    intrusion = 0.0
    required_deceleration = 0.0
    for actor in actors:
        velocity = np.asarray(actor.get("velocity", [0.0, 0.0])[:2], dtype=np.float64)
        actor_speed = float(np.linalg.norm(velocity))
        actor_yaw = float(actor["rotation"][2])
        forward = _rotation(actor_yaw)[:, 0]
        right = _rotation(actor_yaw)[:, 1]
        reaction = actor_speed * cfg_cost.interaction.reaction_time_s \
            + cfg_cost.interaction.min_gap_m
        courtesy = dict(actor)
        courtesy["location"] = list(actor["location"])
        courtesy["extent"] = list(actor["extent"])
        courtesy["location"][0] += forward[0] * reaction * 0.5
        courtesy["location"][1] += forward[1] * reaction * 0.5
        courtesy["extent"][0] += reaction * 0.5
        courtesy["extent"][1] += cfg_cost.interaction.lateral_margin_m
        intrusion = max(intrusion, max(0.0, -_signed_box_distance(box, courtesy)))
        relative = np.asarray(box["location"][:2]) - np.asarray(actor["location"][:2])
        longitudinal = float(np.dot(relative, forward))
        lateral = abs(float(np.dot(relative, right)))
        lateral_limit = actor["extent"][1] + box["extent"][1] \
            + cfg_cost.interaction.lateral_margin_m
        gap = longitudinal - actor["extent"][0] - box["extent"][0]
        if longitudinal > 0.0 and lateral <= lateral_limit:
            denominator = max(gap - cfg_cost.interaction.min_gap_m,
                              np.finfo(np.float64).eps)
            required_deceleration = max(
                required_deceleration, actor_speed ** 2 / (2.0 * denominator))
    values[_TERM_INDEX["interaction.courtesy_intrusion_m"]] = intrusion
    values[_TERM_INDEX["interaction.required_deceleration_mps2"]] = required_deceleration

    previous_arc = float(state.get("previous_route_arc_m", route_arc))
    progress_speed = (route_arc - previous_arc) / float(state["dt_s"])
    relevant = state["relevant_traffic_control"]
    red_reason = relevant.get("valid") and relevant.get("state") == "red" \
        and relevant.get("route_distance", float("inf")) <= cfg_model.stop_red_light_distance_m
    obstacle_reason = _front_obstacle(
        box, actors, cfg_model.stop_obstacle_distance_m,
        cfg_model.stop_obstacle_half_angle_deg)
    reference = min(state["speed_limit_mps"], cfg_cost.efficiency.max_reference_speed_mps)
    values[_TERM_INDEX["efficiency.progress_shortfall_mps"]] = 0.0 if (
        red_reason or obstacle_reason) else max(0.0, reference - progress_speed)
    values[_TERM_INDEX["efficiency.reverse_progress_mps"]] = max(0.0, -progress_speed)
    values[_TERM_INDEX["efficiency.unjustified_stationary_s"]] = float(
        speed < cfg_cost.comfort_control.stationary_speed_mps
        and not red_reason and not obstacle_reason) * float(state["dt_s"])

    dynamics = state["dynamics"]
    for name in ("longitudinal_acceleration_mps2", "lateral_acceleration_mps2",
                 "jerk_mps3", "yaw_rate_radps"):
        values[_TERM_INDEX["comfort_control." + name]] = abs(float(dynamics.get(name, 0.0)))
    control = state.get("control")
    if control is not None and previous_control is not None:
        values[_TERM_INDEX["comfort_control.steer_delta"]] = abs(
            float(control["steer"]) - float(previous_control["steer"]))
        values[_TERM_INDEX["comfort_control.throttle_delta"]] = abs(
            float(control["throttle"]) - float(previous_control["throttle"]))
        values[_TERM_INDEX["comfort_control.brake_delta"]] = abs(
            float(control["brake"]) - float(previous_control["brake"]))
        values[_TERM_INDEX["comfort_control.throttle_brake_overlap"]] = max(
            0.0, float(control["throttle"])) * max(0.0, float(control["brake"]))
    return values, route_arc


def _front_obstacle(ego_box, actors, distance_m, half_angle_deg):
    origin = np.asarray(ego_box["location"][:2], dtype=np.float64)
    yaw = float(ego_box["rotation"][2])
    forward = _rotation(yaw)[:, 0]
    cosine_limit = math.cos(math.radians(float(half_angle_deg)))
    for actor in actors:
        delta = np.asarray(actor["location"][:2], dtype=np.float64) - origin
        norm = float(np.linalg.norm(delta))
        if norm <= np.finfo(np.float64).eps:
            return True
        if float(np.dot(delta / norm, forward)) >= cosine_limit \
                and _signed_box_distance(ego_box, actor) <= distance_m:
            return True
    return False


def _actual_terms(previous, current, cfg_cost, cfg_model, route):
    ego_box = next(box for box in current["bboxes"] if box["semantic"] == "ego")
    current_xy = np.asarray(ego_box["location"][:2], dtype=np.float64)
    previous_box = next(box for box in previous["bboxes"] if box["semantic"] == "ego")
    previous_xy = np.asarray(previous_box["location"][:2], dtype=np.float64)
    velocity = np.asarray(current["ego"]["velocity"][:2], dtype=np.float64)
    speed = float(np.linalg.norm(velocity))
    acceleration = np.asarray(current["ego"]["acceleration"][:2], dtype=np.float64)
    yaw = float(current["ego"]["transform"][5])
    forward, right = _rotation(yaw)[:, 0], _rotation(yaw)[:, 1]
    previous_acceleration = np.asarray(previous["ego"]["acceleration"][:2], dtype=np.float64)
    dt = float(current["sim_time"] - previous["sim_time"])
    state = dict(current)
    state.update({
        "candidate_box": ego_box,
        "dt_s": dt,
        "previous_route_arc_m": float(previous["navigation"]["route_progress_m"]),
        "dynamics": {
            "longitudinal_acceleration_mps2": float(np.dot(acceleration, forward)),
            "lateral_acceleration_mps2": float(np.dot(acceleration, right)),
            "jerk_mps3": float(np.linalg.norm(acceleration - previous_acceleration) / dt),
            "yaw_rate_radps": math.radians(float(current["ego"]["angular_velocity"][2])),
        },
        "control": current["ego"]["control"],
    })
    values, _ = _motion_box(
        state, previous_xy, current_xy, speed, yaw,
        current.get("new_collision_events", 0), previous["ego"]["control"],
        cfg_cost, cfg_model, route)
    return values


def _candidate_terms(step, start_state, future_states, cfg_cost, cfg_model, route):
    trajectories = np.asarray(step["trajectories"], dtype=np.float64)
    modes, points_count = trajectories.shape[:2]
    values = np.zeros((modes, points_count, len(COST_TERMS)), dtype=np.float32)
    valid = np.zeros_like(values, dtype=np.bool_)
    dt = float(step["waypoint_dt_s"])
    base_speed = float(start_state["speed_mps"])
    base_acceleration = float(np.linalg.norm(start_state["ego"]["acceleration"][:2]))
    for mode in range(modes):
        previous_local = np.zeros(2, dtype=np.float64)
        previous_speed = base_speed
        previous_acceleration = base_acceleration
        previous_yaw = 0.0
        previous_arc = float(start_state["navigation"]["route_progress_m"])
        for point_index, future in enumerate(future_states):
            if future is None:
                break
            local = trajectories[mode, point_index]
            delta = local - previous_local
            speed = float(np.linalg.norm(delta) / dt)
            local_yaw = math.degrees(math.atan2(delta[1], delta[0])) \
                if np.linalg.norm(delta) > np.finfo(np.float64).eps else previous_yaw
            acceleration = (speed - previous_speed) / dt
            jerk = (acceleration - previous_acceleration) / dt
            yaw_rate = math.radians(_angle_difference(local_yaw, previous_yaw)) / dt
            box = _ego_box_at(start_state, local, local_yaw)
            world_xy = np.asarray(box["location"][:2], dtype=np.float64)
            previous_world_box = _ego_box_at(start_state, previous_local, previous_yaw)
            candidate_state = dict(future)
            candidate_state.update({
                "candidate_box": box,
                "dt_s": dt,
                "previous_route_arc_m": previous_arc,
                "relevant_traffic_control": start_state["relevant_traffic_control"],
                "speed_limit_mps": start_state["speed_limit_mps"],
                "dynamics": {
                    "longitudinal_acceleration_mps2": acceleration,
                    "lateral_acceleration_mps2": speed * yaw_rate,
                    "jerk_mps3": jerk,
                    "yaw_rate_radps": yaw_rate,
                },
                "control": None,
            })
            result, previous_arc = _motion_box(
                candidate_state, np.asarray(previous_world_box["location"][:2]), world_xy,
                speed, float(start_state["ego"]["transform"][5]) + local_yaw, 0, None,
                cfg_cost, cfg_model, route)
            values[mode, point_index] = result
            valid[mode, point_index] = True
            valid[mode, point_index, list(_ACTUAL_ONLY)] = False
            previous_local, previous_speed = local, speed
            previous_acceleration, previous_yaw = acceleration, local_yaw
    return values, valid


def evaluate_drive_costs(world_states, model_steps, route_geometry, cfg_cost, cfg_model):
    """就地为全部模型步骤添加候选、当前、下一刻和逐项历史代价。"""
    check_cost_inputs(world_states, model_steps, route_geometry)
    ordered = sorted(world_states, key=lambda item: item["frame_id"])
    previous_events = 0
    for state in ordered:
        events = int(state.get("collision_events", 0))
        state["new_collision_events"] = max(events - previous_events, 0)
        previous_events = events
    by_frame = {int(item["frame_id"]): item for item in ordered}
    index_by_frame = {int(item["frame_id"]): index for index, item in enumerate(ordered)}
    route = {
        "points": np.asarray(route_geometry["points"], dtype=np.float64)[:, :2],
        "arc": np.asarray(route_geometry["arc_m"], dtype=np.float64),
        "yaw": np.asarray(route_geometry["yaw_deg"], dtype=np.float64),
        "lane_width": np.asarray(route_geometry["lane_width_m"], dtype=np.float64),
    }
    history = np.zeros(len(COST_TERMS), dtype=np.float64)
    for step in model_steps:
        input_frame = int(step["input_frame_id"])
        next_frame = int(step["next_frame_id"])
        current = by_frame[input_frame]
        following = by_frame[next_frame]
        current_index = index_by_frame[input_frame]
        if current_index > 0:
            previous = ordered[current_index - 1]
        else:
            dt = float(ordered[1]["sim_time"] - current["sim_time"])
            previous = dict(current)
            previous["sim_time"] = float(current["sim_time"] - dt)
        current_terms = _actual_terms(
            previous, current, cfg_cost, cfg_model, route)
        next_terms = _actual_terms(current, following, cfg_cost, cfg_model, route)
        start_index = index_by_frame[input_frame]
        point_count = np.asarray(step["trajectories"]).shape[1]
        futures = [ordered[start_index + offset]
                   if start_index + offset < len(ordered) else None
                   for offset in range(1, point_count + 1)]
        candidate_terms, candidate_valid = _candidate_terms(
            step, current, futures, cfg_cost, cfg_model, route)
        history += next_terms
        step.update({
            "candidate_cost_terms": candidate_terms,
            "candidate_cost_valid": candidate_valid,
            "current_cost_terms": current_terms.astype(np.float32),
            "next_cost_terms": next_terms.astype(np.float32),
            "current_cost_valid": np.ones(len(COST_TERMS), dtype=np.bool_),
            "next_cost_valid": np.ones(len(COST_TERMS), dtype=np.bool_),
            "historical_cost_terms": history.astype(np.float32).copy(),
            "historical_cost_valid": np.ones(len(COST_TERMS), dtype=np.bool_),
        })
