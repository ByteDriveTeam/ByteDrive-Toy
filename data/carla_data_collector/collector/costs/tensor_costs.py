"""Torch 批量候选轨迹代价：GPU/CPU 张量几何与严格包围圆 broad-phase。

模块: collector/costs/tensor_costs.py
依赖: math, numpy, torch
读取配置: 由调用方传入 carla_collector.cost / model_collection
对外接口:
    - evaluate_candidate_costs_torch(...) -> str
说明: 这里只计算可并行的候选轨迹项；真实执行轨迹的少量递推项仍由 costs.py 在 CPU 计算。
      broad-phase 仅排除数学上不可能影响阈值/重叠结果的 OBB 对，不改变代价定义。
"""

import math

import numpy as np
import torch


_EPS = float(np.finfo(np.float64).eps)
_BOX_SIGNS = ((1.0, 1.0), (1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0))


def _axes(yaw):
    """返回局部 x/y 轴，形状 ``[..., 2, 2]``。"""
    cosine, sine = torch.cos(yaw), torch.sin(yaw)
    forward = torch.stack((cosine, sine), dim=-1)
    right = torch.stack((-sine, cosine), dim=-1)
    return torch.stack((forward, right), dim=-2)


def _corners(center, extent, yaw):
    signs = torch.as_tensor(
        _BOX_SIGNS, dtype=center.dtype, device=center.device)
    local = extent.unsqueeze(-2) * signs
    return center.unsqueeze(-2) + torch.einsum(
        "...ki,...id->...kd", local, _axes(yaw))


def _pair_signed_distance(candidate_center, candidate_extent, candidate_yaw,
                          actor_center, actor_extent, actor_yaw):
    """批量精确二维 OBB 有符号距离；输入均为相同首维的 box 对。"""
    candidate_axes = _axes(candidate_yaw)
    actor_axes = _axes(actor_yaw)
    all_axes = torch.cat((candidate_axes, actor_axes), dim=1)
    relative = actor_center - candidate_center

    candidate_projection = torch.einsum(
        "pid,pkd->pki", candidate_axes, all_axes).abs()
    actor_projection = torch.einsum(
        "pid,pkd->pki", actor_axes, all_axes).abs()
    candidate_radius = (candidate_projection * candidate_extent[:, None]).sum(dim=-1)
    actor_radius = (actor_projection * actor_extent[:, None]).sum(dim=-1)
    # 与 CPU 参考实现一致，取两个投影区间的实际交叠长度；当一个区间
    # 完全包含另一个时不能用 ``r1+r2-|dc|``（它会高估穿透深度）。
    center_projection = torch.einsum("pd,pkd->pk", relative, all_axes)
    overlaps = torch.minimum(
        candidate_radius, center_projection + actor_radius) - torch.maximum(
            -candidate_radius, center_projection - actor_radius)
    separated = (overlaps < 0.0).any(dim=-1)
    penetration = -overlaps.min(dim=-1).values

    candidate_corners = _corners(
        candidate_center, candidate_extent, candidate_yaw)
    actor_corners = _corners(actor_center, actor_extent, actor_yaw)
    candidate_in_actor = torch.einsum(
        "pkd,pid->pki", candidate_corners - actor_center[:, None], actor_axes)
    actor_in_candidate = torch.einsum(
        "pkd,pid->pki", actor_corners - candidate_center[:, None], candidate_axes)
    candidate_gap = torch.clamp(
        candidate_in_actor.abs() - actor_extent[:, None], min=0.0)
    actor_gap = torch.clamp(
        actor_in_candidate.abs() - candidate_extent[:, None], min=0.0)
    distance = torch.minimum(
        torch.linalg.vector_norm(candidate_gap, dim=-1).min(dim=-1).values,
        torch.linalg.vector_norm(actor_gap, dim=-1).min(dim=-1).values)
    return torch.where(separated, distance, penetration)


def _sparse_signed_distance(candidate_center, candidate_extent, candidate_yaw,
                            actor_center, actor_extent, actor_yaw, actor_valid,
                            threshold, extra_mask=None):
    """包围圆严格筛选后仅计算可能小于 ``threshold`` 的精确 OBB 对。"""
    relative = actor_center - candidate_center[:, None]
    center_distance = torch.linalg.vector_norm(relative, dim=-1)
    candidate_radius = torch.linalg.vector_norm(candidate_extent, dim=-1)[:, None]
    actor_radius = torch.linalg.vector_norm(actor_extent, dim=-1)
    selected = actor_valid & (
        center_distance - candidate_radius - actor_radius <= float(threshold))
    if extra_mask is not None:
        selected &= extra_mask
    result = torch.full_like(center_distance, float("inf"))
    pair_index = selected.nonzero(as_tuple=False)
    if pair_index.numel() == 0:
        return result
    candidate_index, actor_index = pair_index[:, 0], pair_index[:, 1]
    exact = _pair_signed_distance(
        candidate_center[candidate_index], candidate_extent[candidate_index],
        candidate_yaw[candidate_index],
        actor_center[candidate_index, actor_index],
        actor_extent[candidate_index, actor_index],
        actor_yaw[candidate_index, actor_index])
    result[candidate_index, actor_index] = exact
    return result


def _pack_actor_timeline(ordered):
    actor_lists = [
        [box for box in state["bboxes"] if box["semantic"] != "ego"]
        for state in ordered
    ]
    # 保留一个全 invalid 的占位 actor，使后续 min/max reduction 也覆盖零交通流。
    max_actors = max(max((len(items) for items in actor_lists), default=0), 1)
    shape = (len(ordered), max_actors)
    center = np.zeros(shape + (2,), dtype=np.float32)
    extent = np.zeros(shape + (2,), dtype=np.float32)
    yaw = np.zeros(shape, dtype=np.float32)
    velocity = np.zeros(shape + (2,), dtype=np.float32)
    valid = np.zeros(shape, dtype=np.bool_)
    for state_index, actors in enumerate(actor_lists):
        for actor_index, actor in enumerate(actors):
            center[state_index, actor_index] = actor["location"][:2]
            extent[state_index, actor_index] = actor["extent"][:2]
            yaw[state_index, actor_index] = math.radians(float(actor["rotation"][2]))
            velocity[state_index, actor_index] = actor.get("velocity", [0.0, 0.0])[:2]
            valid[state_index, actor_index] = True
    return center, extent, yaw, velocity, valid


def _pack_start_states(steps, by_frame):
    packed = []
    for step in steps:
        state = by_frame[int(step["input_frame_id"])]
        ego_box = next(box for box in state["bboxes"] if box["semantic"] == "ego")
        relevant = state["relevant_traffic_control"]
        is_red = bool(relevant.get("valid") and relevant.get("state") == "red")
        packed.append({
            "origin": state["ego"]["transform"][:2],
            "base_yaw": math.radians(float(state["ego"]["transform"][5])),
            "box_center": ego_box["location"][:2],
            "box_extent": ego_box["extent"][:2],
            "speed": float(state["speed_mps"]),
            "acceleration": float(np.linalg.norm(state["ego"]["acceleration"][:2])),
            "route_arc": float(state["navigation"]["route_progress_m"]),
            "speed_limit": float(state["speed_limit_mps"]),
            "red": is_red,
            "route_distance": float(relevant.get("route_distance", float("inf"))),
            "stop_location": relevant.get("stop_location", [0.0, 0.0])[:2],
            "stop_yaw": math.radians(float(relevant.get("stop_yaw", 0.0))),
            "lane_width": float(relevant.get("lane_width", 0.0)),
        })
    return packed


def _tensor(items, key, device, dtype=torch.float32):
    return torch.as_tensor(
        np.asarray([item[key] for item in items]), dtype=dtype, device=device)


def _candidate_batch(steps, start_indices, ordered_count, by_frame, actor_timeline,
                     route, cfg_cost, cfg_model, term_index, actual_only, device):
    dtype = torch.float32
    trajectory_array = np.stack([
        np.asarray(step["trajectories"], dtype=np.float32) for step in steps])
    trajectories64 = torch.as_tensor(
        trajectory_array, dtype=torch.float64, device=device)
    trajectories = trajectories64.to(dtype)
    batch, modes, points_count = trajectories.shape[:3]
    starts = _pack_start_states(steps, by_frame)
    dt64 = _tensor(
        steps, "waypoint_dt_s", device, torch.float64)[:, None, None]
    origin = _tensor(starts, "origin", device)
    base_yaw = _tensor(starts, "base_yaw", device)
    box_center = _tensor(starts, "box_center", device)
    box_extent = _tensor(starts, "box_extent", device)
    base_speed = _tensor(
        starts, "speed", device, torch.float64)[:, None, None]
    base_acceleration = _tensor(
        starts, "acceleration", device, torch.float64)[:, None, None]

    previous_local64 = torch.cat((
        torch.zeros_like(trajectories64[:, :, :1]), trajectories64[:, :, :-1]), dim=2)
    delta64 = trajectories64 - previous_local64
    delta_norm64 = torch.linalg.vector_norm(delta64, dim=-1)
    speed = delta_norm64 / dt64
    previous_speed = torch.cat(
        (base_speed.expand(-1, modes, -1), speed[:, :, :-1]), dim=2)
    acceleration = (speed - previous_speed) / dt64
    previous_acceleration = torch.cat(
        (base_acceleration.expand(-1, modes, -1), acceleration[:, :, :-1]), dim=2)
    jerk = (acceleration - previous_acceleration) / dt64

    yaw_steps = []
    previous_yaw = torch.zeros((batch, modes), dtype=dtype, device=device)
    for point_index in range(points_count):
        raw_yaw = torch.atan2(
            delta64[:, :, point_index, 1], delta64[:, :, point_index, 0])
        current_yaw = torch.where(
            delta_norm64[:, :, point_index] > _EPS, raw_yaw, previous_yaw)
        yaw_steps.append(current_yaw)
        previous_yaw = current_yaw
    local_yaw = torch.stack(yaw_steps, dim=2)
    previous_local_yaw = torch.cat(
        (torch.zeros_like(local_yaw[:, :, :1]), local_yaw[:, :, :-1]), dim=2)
    yaw_delta = torch.remainder(
        local_yaw - previous_local_yaw + math.pi, 2.0 * math.pi) - math.pi
    yaw_rate = yaw_delta.abs() / dt64
    local_yaw_geometry = local_yaw.to(dtype)

    base_axes = _axes(base_yaw)
    world_xy = origin[:, None, None] + torch.einsum(
        "bmti,bid->bmtd", trajectories, base_axes)
    offset = torch.einsum(
        "bd,bid->bi", box_center - origin, base_axes)
    world_yaw = base_yaw[:, None, None] + local_yaw_geometry
    candidate_center = world_xy + torch.einsum(
        "bi,bmtid->bmtd", offset, _axes(world_yaw))
    candidate_extent = box_extent[:, None, None].expand(-1, modes, points_count, -1)
    initial_center = box_center[:, None, None].expand(-1, modes, 1, -1)
    previous_center = torch.cat((initial_center, candidate_center[:, :, :-1]), dim=2)

    future_offsets = torch.arange(1, points_count + 1, device=device)
    future_indices = torch.as_tensor(start_indices, device=device)[:, None] + future_offsets
    future_valid = future_indices < ordered_count
    future_indices = future_indices.clamp(max=max(ordered_count - 1, 0))
    actor_center_all, actor_extent_all, actor_yaw_all, actor_velocity_all, actor_valid_all = \
        actor_timeline
    actor_center = actor_center_all[future_indices]
    actor_extent = actor_extent_all[future_indices]
    actor_yaw = actor_yaw_all[future_indices]
    actor_velocity = actor_velocity_all[future_indices]
    actor_valid = actor_valid_all[future_indices] & future_valid[:, :, None]

    flat_count = batch * modes * points_count
    flat_center = candidate_center.reshape(flat_count, 2)
    flat_extent = candidate_extent.reshape(flat_count, 2)
    flat_yaw = world_yaw.reshape(flat_count)
    expand_shape = (batch, modes, points_count, actor_center.shape[2])
    flat_actor_center = actor_center[:, None].expand(expand_shape + (2,)).reshape(
        flat_count, actor_center.shape[2], 2)
    flat_actor_extent = actor_extent[:, None].expand(expand_shape + (2,)).reshape(
        flat_count, actor_extent.shape[2], 2)
    flat_actor_yaw = actor_yaw[:, None].expand(expand_shape).reshape(
        flat_count, actor_yaw.shape[2])
    flat_actor_velocity = actor_velocity[:, None].expand(expand_shape + (2,)).reshape(
        flat_count, actor_velocity.shape[2], 2)
    flat_actor_valid = actor_valid[:, None].expand(expand_shape).reshape(
        flat_count, actor_valid.shape[2])

    term_count = len(term_index)
    values = torch.zeros((flat_count, term_count), dtype=dtype, device=device)
    safety_distance = _sparse_signed_distance(
        flat_center, flat_extent, flat_yaw, flat_actor_center, flat_actor_extent,
        flat_actor_yaw, flat_actor_valid, cfg_cost.safety.safe_clearance_m)
    minimum = safety_distance.min(dim=-1).values
    values[:, term_index["safety.clearance_deficit_m"]] = torch.clamp(
        float(cfg_cost.safety.safe_clearance_m) - minimum, min=0.0)
    values[:, term_index["safety.overlap_depth_m"]] = torch.clamp(-minimum, min=0.0)

    candidate_corners = _corners(flat_center, flat_extent, flat_yaw)
    route_points = route["points"]
    queries = torch.cat((candidate_corners, flat_center[:, None]), dim=1)
    route_distance = torch.cdist(
        queries.reshape(-1, 2), route_points).reshape(flat_count, 5, -1)
    route_indices = route_distance.argmin(dim=-1)
    nearest_distance = route_distance.gather(-1, route_indices[..., None]).squeeze(-1)
    corner_indices, center_indices = route_indices[:, :4], route_indices[:, 4]
    corner_lane_width = route["lane_width"][corner_indices]
    overflow = torch.clamp(
        nearest_distance[:, :4] - corner_lane_width * 0.5
        - float(cfg_cost.compliance.route_margin_m), min=0.0)
    values[:, term_index["compliance.route_overflow_m"]] = overflow.max(dim=-1).values
    route_yaw = route["yaw"][center_indices]
    wrong_way = torch.remainder(flat_yaw - route_yaw + math.pi, 2.0 * math.pi) - math.pi
    values[:, term_index["compliance.wrong_way_excess_deg"]] = torch.clamp(
        torch.rad2deg(wrong_way.abs())
        - float(cfg_cost.compliance.wrong_way_tolerance_deg), min=0.0)
    flat_speed = speed.reshape(flat_count)
    speed_limit = _tensor(
        starts, "speed_limit", device, torch.float64)[:, None, None].expand(
        -1, modes, points_count).reshape(flat_count)
    values[:, term_index["compliance.overspeed_excess_mps"]] = torch.clamp(
        flat_speed - speed_limit - float(cfg_cost.compliance.speed_tolerance_mps), min=0.0)

    red = _tensor(starts, "red", device, torch.bool)[:, None, None].expand(
        -1, modes, points_count).reshape(flat_count)
    stop_center = _tensor(starts, "stop_location", device)[:, None, None].expand(
        -1, modes, points_count, -1).reshape(flat_count, 2)
    stop_yaw = _tensor(starts, "stop_yaw", device)[:, None, None].expand(
        -1, modes, points_count).reshape(flat_count)
    lane_width = _tensor(starts, "lane_width", device)[:, None, None].expand(
        -1, modes, points_count).reshape(flat_count)
    lateral = torch.stack((-torch.sin(stop_yaw), torch.cos(stop_yaw)), dim=-1)
    stop_a = stop_center - lateral * lane_width[:, None] * 0.5
    stop_b = stop_center + lateral * lane_width[:, None] * 0.5
    motion = flat_center - previous_center.reshape(flat_count, 2)
    stop_line = stop_b - stop_a
    denominator = motion[:, 0] * stop_line[:, 1] - motion[:, 1] * stop_line[:, 0]
    delta_stop = stop_a - previous_center.reshape(flat_count, 2)
    safe_denominator = torch.where(denominator.abs() > _EPS, denominator,
                                   torch.ones_like(denominator))
    intersection_t = (delta_stop[:, 0] * stop_line[:, 1]
                      - delta_stop[:, 1] * stop_line[:, 0]) / safe_denominator
    intersection_u = (delta_stop[:, 0] * motion[:, 1]
                      - delta_stop[:, 1] * motion[:, 0]) / safe_denominator
    crossing = red & (denominator.abs() > _EPS) \
        & (intersection_t >= 0.0) & (intersection_t <= 1.0) \
        & (intersection_u >= 0.0) & (intersection_u <= 1.0)
    values[:, term_index["compliance.red_light_crossings"]] = crossing.to(dtype)

    actor_speed = torch.linalg.vector_norm(flat_actor_velocity, dim=-1)
    actor_axes = _axes(flat_actor_yaw)
    actor_forward, actor_right = actor_axes[:, :, 0], actor_axes[:, :, 1]
    reaction = actor_speed * float(cfg_cost.interaction.reaction_time_s) \
        + float(cfg_cost.interaction.min_gap_m)
    courtesy_center = flat_actor_center + actor_forward * reaction[..., None] * 0.5
    courtesy_extent = flat_actor_extent.clone()
    courtesy_extent[:, :, 0] += reaction * 0.5
    courtesy_extent[:, :, 1] += float(cfg_cost.interaction.lateral_margin_m)
    courtesy_distance = _sparse_signed_distance(
        flat_center, flat_extent, flat_yaw, courtesy_center, courtesy_extent,
        flat_actor_yaw, flat_actor_valid, 0.0)
    values[:, term_index["interaction.courtesy_intrusion_m"]] = torch.clamp(
        -courtesy_distance, min=0.0).max(dim=-1).values

    relative = flat_center[:, None] - flat_actor_center
    longitudinal = (relative * actor_forward).sum(dim=-1)
    lateral_distance = (relative * actor_right).sum(dim=-1).abs()
    lateral_limit = flat_actor_extent[:, :, 1] + flat_extent[:, None, 1] \
        + float(cfg_cost.interaction.lateral_margin_m)
    gap = longitudinal - flat_actor_extent[:, :, 0] - flat_extent[:, None, 0]
    denominator_gap = torch.clamp(
        gap - float(cfg_cost.interaction.min_gap_m), min=_EPS)
    required = actor_speed.square() / (2.0 * denominator_gap)
    required_mask = flat_actor_valid & (longitudinal > 0.0) \
        & (lateral_distance <= lateral_limit)
    required = torch.where(required_mask, required, torch.zeros_like(required))
    values[:, term_index["interaction.required_deceleration_mps2"]] = \
        required.max(dim=-1).values

    ego_forward = _axes(flat_yaw)[:, 0]
    actor_delta = flat_actor_center - flat_center[:, None]
    actor_norm = torch.linalg.vector_norm(actor_delta, dim=-1)
    normalized_delta = actor_delta / torch.clamp(actor_norm[..., None], min=_EPS)
    cosine_limit = math.cos(math.radians(float(cfg_model.stop_obstacle_half_angle_deg)))
    in_front = (normalized_delta * ego_forward[:, None]).sum(dim=-1) >= cosine_limit
    same_center = flat_actor_valid & (actor_norm <= _EPS)
    front_distance = _sparse_signed_distance(
        flat_center, flat_extent, flat_yaw, flat_actor_center, flat_actor_extent,
        flat_actor_yaw, flat_actor_valid, cfg_model.stop_obstacle_distance_m,
        extra_mask=in_front)
    obstacle = same_center.any(dim=-1) | (
        front_distance <= float(cfg_model.stop_obstacle_distance_m)).any(dim=-1)

    route_arc = route["arc"][center_indices].reshape(batch, modes, points_count)
    starting_arc = _tensor(
        starts, "route_arc", device, torch.float64)[:, None, None].expand(
        -1, modes, 1)
    previous_arc = torch.cat((starting_arc, route_arc[:, :, :-1]), dim=2)
    progress_speed = ((route_arc - previous_arc) / dt64).reshape(flat_count)
    route_distance_to_light = _tensor(starts, "route_distance", device)
    red_reason = (_tensor(starts, "red", device, torch.bool)
                  & (route_distance_to_light
                     <= float(cfg_model.stop_red_light_distance_m)))
    red_reason = red_reason[:, None, None].expand(
        -1, modes, points_count).reshape(flat_count)
    reference = torch.minimum(
        speed_limit,
        torch.full_like(speed_limit, float(cfg_cost.efficiency.max_reference_speed_mps)))
    stop_reason = red_reason | obstacle
    values[:, term_index["efficiency.progress_shortfall_mps"]] = torch.where(
        stop_reason, torch.zeros_like(reference),
        torch.clamp(reference - progress_speed, min=0.0))
    values[:, term_index["efficiency.reverse_progress_mps"]] = torch.clamp(
        -progress_speed, min=0.0)
    values[:, term_index["efficiency.unjustified_stationary_s"]] = (
        (flat_speed < float(cfg_cost.comfort_control.stationary_speed_mps))
        & ~stop_reason).to(dtype) * dt64.expand(-1, modes, points_count).reshape(flat_count)

    values[:, term_index["comfort_control.longitudinal_acceleration_mps2"]] = \
        acceleration.abs().reshape(flat_count)
    values[:, term_index["comfort_control.lateral_acceleration_mps2"]] = \
        (speed * yaw_rate).abs().reshape(flat_count)
    values[:, term_index["comfort_control.jerk_mps3"]] = jerk.abs().reshape(flat_count)
    values[:, term_index["comfort_control.yaw_rate_radps"]] = \
        yaw_rate.abs().reshape(flat_count)

    values = values.reshape(batch, modes, points_count, term_count)
    valid = future_valid[:, None, :, None].expand(
        -1, modes, -1, term_count).clone()
    valid[:, :, :, list(actual_only)] = False
    values *= future_valid[:, None, :, None]
    return values.cpu().numpy(), valid.cpu().numpy()


def _resolve_device(requested):
    requested = str(requested or "cuda")
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def evaluate_candidate_costs_torch(ordered, model_steps, by_frame, index_by_frame,
                                   route, cfg_cost, cfg_model, term_index, actual_only,
                                   requested_device=None, batch_steps=None):
    """批量计算并就地写入全部 step 的候选代价，返回实际 Torch 设备字符串。"""
    device = _resolve_device(requested_device)
    if batch_steps is None:
        batch_steps = 32 if device.type == "cuda" else 4
    actor_arrays = _pack_actor_timeline(ordered)
    actor_timeline = tuple(torch.as_tensor(item, device=device) for item in actor_arrays)
    tensor_route = {
        "points": torch.as_tensor(route["points"], dtype=torch.float32, device=device),
        "arc": torch.as_tensor(route["arc"], dtype=torch.float64, device=device),
        "yaw": torch.deg2rad(torch.as_tensor(
            route["yaw"], dtype=torch.float32, device=device)),
        "lane_width": torch.as_tensor(
            route["lane_width"], dtype=torch.float32, device=device),
    }

    cursor = 0
    with torch.inference_mode():
        while cursor < len(model_steps):
            shape = np.asarray(model_steps[cursor]["trajectories"]).shape
            end = cursor
            while end < len(model_steps) and end - cursor < batch_steps \
                    and np.asarray(model_steps[end]["trajectories"]).shape == shape:
                end += 1
            chunk = model_steps[cursor:end]
            indices = [index_by_frame[int(step["input_frame_id"])] for step in chunk]
            try:
                values, valid = _candidate_batch(
                    chunk, indices, len(ordered), by_frame, actor_timeline,
                    tensor_route, cfg_cost, cfg_model, term_index, actual_only, device)
            except torch.cuda.OutOfMemoryError:
                if device.type != "cuda":
                    raise
                torch.cuda.empty_cache()
                if batch_steps > 1:
                    batch_steps = max(batch_steps // 2, 1)
                    continue
                # 极端交通密度下单步仍放不进显存时，仅把剩余候选回退到
                # Torch CPU；已完成的 GPU 批次无需重算，数值与格式保持一致。
                device = torch.device("cpu")
                actor_timeline = tuple(item.cpu() for item in actor_timeline)
                tensor_route = {key: value.cpu() for key, value in tensor_route.items()}
                batch_steps = 4
                print("[collector] 候选代价单步超出 CUDA 显存，剩余批次回退 Torch CPU")
                continue
            for item, item_values, item_valid in zip(chunk, values, valid):
                item["candidate_cost_terms"] = item_values
                item["candidate_cost_valid"] = item_valid
            cursor = end
    return str(device)
