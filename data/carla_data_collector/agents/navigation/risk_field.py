"""Route-aligned dynamic risk field built from Bayesian actor predictions."""

import math
from dataclasses import replace

import numpy as np


def _interval_probability(mean, sigma, half_width):
    """Probability that a Gaussian center falls inside a physical corridor."""
    scale = max(float(sigma), 1e-3) * math.sqrt(2.0)
    upper = (float(half_width) - float(mean)) / scale
    lower = (-float(half_width) - float(mean)) / scale
    return float(np.clip(0.5 * (math.erf(upper) - math.erf(lower)), 0.0, 1.0))


def _route_geometry(route_points):
    points = np.asarray(route_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 2:
        return None
    points = points[:, :2]
    if len(points) == 1:
        return points, np.zeros((1,), dtype=np.float64), np.zeros((1, 2)), np.zeros((1,))
    delta = np.diff(points, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    tangent = np.zeros_like(points)
    tangent[1:] = delta / np.maximum(lengths[:, None], 1e-6)
    tangent[0] = tangent[1]
    arc = np.concatenate(([0.0], np.cumsum(lengths)))
    return points, arc, tangent, lengths


def _nearest_route_point(position, points, arc, tangent):
    if hasattr(position, "x") and hasattr(position, "y"):
        position = [position.x, position.y]
    delta = points - np.asarray(position, dtype=np.float64)[:2]
    distance = np.linalg.norm(delta, axis=1)
    index = int(np.argmin(distance))
    forward = float(-np.dot(delta[index], tangent[index]))
    # Preserve a negative station for actors behind the route origin.  Clamping
    # it to zero makes a following vehicle overlap the ego in route space.
    return index, float(distance[index]), float(arc[index] + forward)


def _project_route_segment(position, points, arc):
    """Project a position onto route segments, including beyond the endpoints."""
    if hasattr(position, "x") and hasattr(position, "y"):
        position = [position.x, position.y]
    position = np.asarray(position, dtype=np.float64)[:2]
    if len(points) < 2:
        return 0.0, float(np.linalg.norm(position - points[0])), np.zeros(2)
    segments = points[1:] - points[:-1]
    lengths_sq = np.sum(segments * segments, axis=1)
    raw = np.sum((position - points[:-1]) * segments, axis=1) / np.maximum(
        lengths_sq, 1e-9)
    clipped = np.clip(raw, 0.0, 1.0)
    nearest = points[:-1] + clipped[:, None] * segments
    index = int(np.argmin(np.sum((nearest - position) ** 2, axis=1)))
    fraction = float(raw[index]) if index in (0, len(segments) - 1) else float(clipped[index])
    length = math.sqrt(max(float(lengths_sq[index]), 1e-9))
    direction = segments[index] / length
    normal = np.asarray([-direction[1], direction[0]])
    station = float(arc[index] + fraction * length)
    lateral = float(np.dot(position - points[index], normal))
    return station, lateral, direction


def _vehicle_route_extents(prediction, route_direction):
    """Project an oriented vehicle box using its predicted travel direction."""
    length = max(float(prediction.extent_xy[0]), 0.5)
    width = max(float(prediction.extent_xy[1]), 0.25)
    second = 1 if len(prediction.means) > 1 else 0
    motion = np.asarray(prediction.means[second], dtype=np.float64)[:2] - np.asarray(
        prediction.means[0], dtype=np.float64)[:2]
    magnitude = float(np.linalg.norm(motion))
    if magnitude < 0.25:
        motion = (np.asarray(prediction.means[-1], dtype=np.float64)[:2] -
                  np.asarray(prediction.means[0], dtype=np.float64)[:2])
        magnitude = float(np.linalg.norm(motion))
    if magnitude < 0.25:
        return length, width, 1.0
    heading = motion / magnitude
    heading_side = np.asarray([-heading[1], heading[0]])
    aligned = abs(float(np.dot(heading, route_direction)))
    crossed = abs(float(np.dot(heading_side, route_direction)))
    return (aligned * length + crossed * width,
            crossed * length + aligned * width, aligned)


def plan_front_vehicle_speed(route_points, predictions, ego_position,
                             ego_speed_mps, ego_half_length_m,
                             ego_half_width_m, lateral_margin_m,
                             reaction_time_s, comfort_decel_mps2,
                             minimum_gap_m=3.0, confidence_z=1.0,
                             lookahead_s=1.5, emergency_decel_mps2=8.0):
    """Cap speed using the front vehicle's Bayesian route trajectory.

    The current physical gap supplies the braking constraint. Near-term
    posterior means and longitudinal covariance account for a lead vehicle
    that is slowing or whose future position is uncertain. Stable vehicles in
    adjacent lanes are excluded by their lateral occupancy probability.
    """
    predictions = tuple(predictions or ())
    if any(getattr(item, "modes", ()) for item in predictions):
        unbounded = max(float(ego_speed_mps), 0.0) + 5.0

        def single(item):
            return plan_front_vehicle_speed(
                route_points, [item], ego_position, ego_speed_mps,
                ego_half_length_m, ego_half_width_m, lateral_margin_m,
                reaction_time_s, comfort_decel_mps2,
                minimum_gap_m=minimum_gap_m, confidence_z=confidence_z,
                lookahead_s=lookahead_s,
                emergency_decel_mps2=emergency_decel_mps2)

        result = plan_front_vehicle_speed(
            route_points, [replace(item, modes=()) for item in predictions],
            ego_position, ego_speed_mps, ego_half_length_m,
            ego_half_width_m, lateral_margin_m, reaction_time_s,
            comfort_decel_mps2, minimum_gap_m=minimum_gap_m,
            confidence_z=confidence_z, lookahead_s=lookahead_s,
            emergency_decel_mps2=emergency_decel_mps2)
        for prediction in predictions:
            modes = getattr(prediction, "modes", ())
            if not modes:
                continue
            weighted_cap = 0.0
            total_probability = 0.0
            crossing_probability = 0.0
            representative = None
            for mode in modes:
                probability = max(float(mode.probability), 0.0)
                mode_plan = single(prediction.for_mode(mode))
                weighted_cap += probability * min(
                    mode_plan["speed_cap_mps"], unbounded)
                total_probability += probability
                if (mode_plan["kind"] == "crossing" and
                        math.isfinite(mode_plan["speed_cap_mps"])):
                    crossing_probability += probability
                if (representative is None or
                        mode_plan["speed_cap_mps"] <
                        representative["speed_cap_mps"]):
                    representative = mode_plan
                if probability >= 0.5 and mode_plan["emergency"]:
                    result["emergency"] = True
            if total_probability <= 0.0:
                continue
            mixed_cap = weighted_cap / total_probability
            if (mixed_cap < result["speed_cap_mps"] and
                    mixed_cap < unbounded - 1e-6 and
                    representative["actor_id"] is not None):
                emergency = result["emergency"]
                existing_crossing_constraint = result.get(
                    "crossing_constraint", result["kind"] == "crossing")
                result = dict(representative)
                result["speed_cap_mps"] = mixed_cap
                result["emergency"] = bool(emergency)
                result["crossing_constraint"] = bool(
                    existing_crossing_constraint or
                    crossing_probability / total_probability >= 0.5)
        return result
    result = {"speed_cap_mps": float("inf"), "emergency": False,
              "actor_id": None, "gap_m": float("inf"), "kind": None}
    geometry = _route_geometry(route_points)
    if geometry is None or len(geometry[0]) < 2:
        return result
    points, arc, _, _ = geometry
    ego_arc, _, ego_direction = _project_route_segment(ego_position, points, arc)
    ego_xy = (np.asarray([ego_position.x, ego_position.y], dtype=np.float64)
              if hasattr(ego_position, "x") else
              np.asarray(ego_position, dtype=np.float64)[:2])
    ego_speed = max(float(ego_speed_mps), 0.0)
    decel = max(float(comfort_decel_mps2), 0.1)
    for prediction in predictions or ():
        if prediction.actor_type != "vehicle" or len(prediction.means) == 0:
            continue
        first_arc, first_lateral, first_direction = _project_route_segment(
            prediction.means[0], points, arc)
        extent_x, extent_y, alignment = _vehicle_route_extents(
            prediction, first_direction)
        # A route that curves back can project a physically trailing actor onto
        # an upcoming segment. It cannot require ego braking while it remains
        # well behind the ego's current heading.
        relative_along = float(np.dot(
            np.asarray(prediction.means[0], dtype=np.float64)[:2] - ego_xy,
            ego_direction))
        if relative_along < -(max(float(ego_half_length_m), 0.5) + extent_x + 1.0):
            continue
        corridor = max(float(ego_half_width_m), 0.5) + extent_y + max(
            float(lateral_margin_m), 0.0)
        first_cov = np.asarray(prediction.covariances[0], dtype=np.float64)
        first_normal = np.asarray([-first_direction[1], first_direction[0]])
        first_sigma = math.sqrt(max(float(first_normal.dot(first_cov).dot(
            first_normal)), 0.05))
        initially_in_route = (alignment >= 0.6 and
                              abs(first_lateral) <= corridor and
                              _interval_probability(
                                  first_lateral, first_sigma, corridor) >= 0.5)
        if not initially_in_route:
            # A side vehicle matters when its *mean* enters the route while
            # the ego would still occupy the crossing. The mean requirement
            # keeps a parallel neighbor's growing covariance from braking us.
            if ego_speed < 1.0:
                continue
            closest_mean_lateral = min(
                abs(_project_route_segment(mean, points, arc)[1])
                for mean in prediction.means)
            # A long vehicle in the next lane can touch an inflated corridor
            # without moving into our path. Require actual inward progress.
            if (abs(first_lateral) > corridor and
                    abs(first_lateral) - closest_mean_lateral < 0.75):
                continue
            for mean, covariance, time_s in zip(
                    prediction.means, prediction.covariances, prediction.times):
                time_s = float(time_s)
                actor_arc, lateral, direction = _project_route_segment(
                    mean, points, arc)
                step_extent_x, step_extent_y, _ = _vehicle_route_extents(
                    prediction, direction)
                step_corridor = (max(float(ego_half_width_m), 0.5) +
                                 step_extent_y + max(float(lateral_margin_m), 0.0))
                if abs(lateral) > step_corridor or actor_arc <= ego_arc + 0.5:
                    continue
                entry_time = (actor_arc - ego_arc -
                              max(float(ego_half_length_m), 0.5) - step_extent_x) / ego_speed
                exit_time = (actor_arc - ego_arc +
                             max(float(ego_half_length_m), 0.5) + step_extent_x) / ego_speed
                if time_s < entry_time - 0.35 or time_s > exit_time + 0.35:
                    continue
                normal = np.asarray([-direction[1], direction[0]])
                covariance = np.asarray(covariance, dtype=np.float64)
                sigma_lateral = math.sqrt(max(float(normal.dot(
                    covariance).dot(normal)), 0.05))
                if _interval_probability(lateral, sigma_lateral,
                                         step_corridor) < 0.35:
                    continue
                sigma_longitudinal = math.sqrt(max(float(direction.dot(
                    covariance).dot(direction)), 0.05))
                current_gap = (actor_arc - ego_arc -
                               max(float(ego_half_length_m), 0.5) - step_extent_x)
                lower_gap = current_gap - max(float(confidence_z), 0.0) * min(
                    sigma_longitudinal, max(float(ego_half_length_m), 0.5))
                usable_gap = max(lower_gap - max(float(minimum_gap_m), 0.0) -
                                 max(float(reaction_time_s), 0.0) * ego_speed,
                                 0.0)
                speed_cap = math.sqrt(2.0 * decel * usable_gap)
                hard_gap = (max(float(minimum_gap_m), 0.0) + 0.3 * ego_speed +
                            ego_speed ** 2 /
                            (2.0 * max(float(emergency_decel_mps2), decel)))
                emergency = (current_gap < hard_gap or
                             (speed_cap <= 0.5 and ego_speed > 1.0))
                if speed_cap < result["speed_cap_mps"]:
                    result = {"speed_cap_mps": float(speed_cap),
                              "emergency": bool(emergency),
                              "actor_id": int(prediction.actor_id),
                              "gap_m": float(current_gap), "kind": "crossing"}
                elif emergency:
                    result["emergency"] = True
            continue
        first_time = float(prediction.times[0])
        if len(prediction.means) > 1:
            second_arc, _, _ = _project_route_segment(
                prediction.means[1], points, arc)
            second_time = float(prediction.times[1])
            lead_speed = (second_arc - first_arc) / max(
                second_time - first_time, 1e-3)
        else:
            lead_speed = float(prediction.speed_mps)
        lead_speed = max(0.0, min(float(lead_speed),
                                  max(float(prediction.speed_mps), 0.0) + 1.5))
        current_arc = first_arc - lead_speed * first_time
        if current_arc <= ego_arc + 0.5:
            continue
        gap = current_arc - ego_arc - max(float(ego_half_length_m), 0.5) - extent_x
        effective_gap = gap
        for mean, covariance, time_s in zip(
                prediction.means, prediction.covariances, prediction.times):
            time_s = float(time_s)
            if time_s > lookahead_s + 1e-6:
                break
            actor_arc, lateral, direction = _project_route_segment(
                mean, points, arc)
            normal = np.asarray([-direction[1], direction[0]])
            covariance = np.asarray(covariance, dtype=np.float64)
            lateral_sigma = math.sqrt(max(float(normal.dot(covariance).dot(
                normal)), 0.05))
            if _interval_probability(lateral, lateral_sigma, corridor) < 0.5:
                continue
            longitudinal_sigma = math.sqrt(max(float(direction.dot(
                covariance).dot(direction)), 0.05))
            lower_arc = actor_arc - max(float(confidence_z), 0.0) * longitudinal_sigma
            expected_arc = current_arc + lead_speed * time_s
            effective_gap = min(effective_gap, gap + lower_arc - expected_arc)
        usable_gap = max(effective_gap - max(float(minimum_gap_m), 0.0) -
                         max(float(reaction_time_s), 0.0) * ego_speed, 0.0)
        speed_cap = math.sqrt(lead_speed ** 2 + 2.0 * decel * usable_gap)
        hard_gap = (max(float(minimum_gap_m), 0.0) + 0.3 * ego_speed +
                    max(ego_speed ** 2 - lead_speed ** 2, 0.0) /
                    (2.0 * max(float(emergency_decel_mps2), decel)))
        emergency = gap < hard_gap and ego_speed > lead_speed + 0.8
        if speed_cap < result["speed_cap_mps"]:
            result = {"speed_cap_mps": float(speed_cap),
                      "emergency": bool(emergency),
                      "actor_id": int(prediction.actor_id),
                      "gap_m": float(gap), "kind": "following"}
        elif emergency:
            result["emergency"] = True
    return result


def evaluate_route_risk(route_points, predictions, ego_position, ego_speed_mps,
                        route_width_m=2.0, risk_decay_m=35.0,
                        pedestrian_route_width_m=None):
    """Return scalar risk and interpretable details for a route corridor.

    ``predictions`` are :class:`ActorPrediction` objects.  The returned values
    are plain Python types so they can be written directly by msgpack.
    """
    predictions = tuple(predictions or ())
    if any(getattr(item, "modes", ()) for item in predictions):
        best = evaluate_route_risk(
            route_points, [replace(item, modes=()) for item in predictions],
            ego_position, ego_speed_mps, route_width_m=route_width_m,
            risk_decay_m=risk_decay_m,
            pedestrian_route_width_m=pedestrian_route_width_m)
        max_probability = best["predicted_conflict_probability"]
        expected_ttc = best["expected_ttc_s"]

        def single(item):
            return evaluate_route_risk(
                route_points, [item], ego_position, ego_speed_mps,
                route_width_m=route_width_m, risk_decay_m=risk_decay_m,
                pedestrian_route_width_m=pedestrian_route_width_m)

        for prediction in predictions:
            modes = getattr(prediction, "modes", ())
            if not modes:
                continue
            weighted_risk = 0.0
            weighted_probability = 0.0
            weighted_ttc = float("inf")
            representative = None
            strongest = 0.0
            total_probability = 0.0
            for mode in modes:
                probability = max(float(mode.probability), 0.0)
                mode_risk = single(prediction.for_mode(mode))
                weighted_risk += probability * mode_risk["max_risk"]
                weighted_probability += probability * mode_risk[
                    "predicted_conflict_probability"]
                total_probability += probability
                contribution = probability * mode_risk["max_risk"]
                if contribution > strongest:
                    strongest = contribution
                    representative = mode_risk
                if probability * mode_risk[
                        "predicted_conflict_probability"] >= 0.25:
                    weighted_ttc = min(
                        weighted_ttc, mode_risk["expected_ttc_s"])
            if total_probability <= 0.0:
                continue
            weighted_risk /= total_probability
            weighted_probability /= total_probability
            max_probability = max(max_probability, weighted_probability)
            expected_ttc = min(expected_ttc, weighted_ttc)
            if weighted_risk > best["max_risk"] and representative is not None:
                best = dict(representative)
                best["max_risk"] = weighted_risk
        best["predicted_conflict_probability"] = max_probability
        best["expected_ttc_s"] = expected_ttc
        best["active"] = bool(best["max_risk"] > 0.05)
        return best
    geometry = _route_geometry(route_points)
    if geometry is None:
        return {
            "max_risk": 0.0, "expected_ttc_s": float("inf"),
            "closest_actor_id": None, "closest_actor_type": None,
            "longitudinal_gap_m": float("inf"), "lateral_gap_m": float("inf"),
            "predicted_conflict_probability": 0.0, "active": False,
        }
    points, arc, tangent, _ = geometry
    _, ego_distance, ego_arc = _nearest_route_point(ego_position, points, arc, tangent)
    max_risk = 0.0
    max_probability = 0.0
    closest = None
    expected_ttc = float("inf")
    closest_gap = float("inf")
    closest_lateral = float("inf")
    for prediction in predictions or ():
        if len(prediction.means) == 0:
            continue
        # Longitudinal braking cannot resolve a conflict caused by a vehicle
        # already following the ego.  Do not let an aggressive extrapolation
        # from behind turn it into a front obstacle for this decision cycle.
        _, _, initial_arc = _nearest_route_point(
            prediction.means[0], points, arc, tangent)
        if prediction.actor_type == "vehicle" and initial_arc < ego_arc - 0.5:
            continue
        extent = max(prediction.extent_xy[1], 0.25)
        width = (pedestrian_route_width_m
                 if prediction.actor_type == "pedestrian" and
                 pedestrian_route_width_m is not None else route_width_m)
        corridor = float(width) + extent
        # Uncertainty growth alone should not make an adjacent-lane vehicle
        # occupy the ego's path. Its mean must approach the physical corridor;
        # a forecast crossing into the corridor still keeps full risk.
        mean_lateral_offsets = []
        for mean in prediction.means:
            nearest, _, _ = _nearest_route_point(mean, points, arc, tangent)
            normal = np.asarray([-tangent[nearest, 1], tangent[nearest, 0]])
            mean_lateral_offsets.append(abs(float(np.dot(
                np.asarray(mean[:2], dtype=np.float64) - points[nearest],
                normal))))
        min_mean_lateral = min(mean_lateral_offsets, default=float("inf"))
        outside_fraction = float(np.clip(
            (min_mean_lateral - corridor) / 0.2, 0.0, 1.0))
        smooth_outside = outside_fraction ** 2 * (3.0 - 2.0 * outside_fraction)
        non_crossing_prior = (
            1.0 - 0.9 * smooth_outside
            if prediction.actor_type == "vehicle" else 1.0)
        for step, (mean, covariance) in enumerate(zip(
                prediction.means, prediction.covariances)):
            index, distance, actor_arc = _nearest_route_point(mean, points, arc, tangent)
            time_s = float(prediction.times[min(step, len(prediction.times) - 1)])
            longitudinal = actor_arc - ego_arc
            if longitudinal < -2.0:
                continue
            lateral = distance
            if lateral > corridor + 4.0:
                continue
            tangent_vec = tangent[index]
            normal = np.asarray([-tangent_vec[1], tangent_vec[0]])
            rel = np.asarray(mean[:2], dtype=np.float64) - points[index]
            lateral_signed = float(np.dot(rel, normal))
            variance = max(float(normal.dot(covariance).dot(normal)), 0.05)
            longitudinal_variance = max(float(tangent_vec.dot(covariance).dot(tangent_vec)), 0.05)
            sigma_lateral = math.sqrt(variance)
            sigma_longitudinal = math.sqrt(longitudinal_variance)
            lateral_probability = _interval_probability(
                lateral_signed, sigma_lateral, corridor)
            future_gap = longitudinal - max(float(ego_speed_mps), 0.0) * time_s
            longitudinal_probability = _interval_probability(
                future_gap, sigma_longitudinal,
                max(prediction.extent_xy[0], 0.5) + 2.2)
            probability = float(np.clip(
                lateral_probability * longitudinal_probability * non_crossing_prior,
                0.0, 1.0))
            distance_weight = math.exp(-max(longitudinal, 0.0) / max(risk_decay_m, 1e-3))
            actor_weight = 1.15 if prediction.actor_type == "pedestrian" else 1.0
            risk = float(np.clip(probability * distance_weight * actor_weight, 0.0, 1.0))
            if risk > max_risk:
                max_risk = risk
                closest_gap = longitudinal
                closest_lateral = abs(lateral_signed)
                closest = prediction
            max_probability = max(max_probability, probability)
            if probability > 0.25 and longitudinal >= 0:
                expected_ttc = min(expected_ttc, time_s)
    return {
        "max_risk": float(max_risk),
        "expected_ttc_s": float(expected_ttc),
        "closest_actor_id": int(closest.actor_id) if closest is not None else None,
        "closest_actor_type": closest.actor_type if closest is not None else None,
        "longitudinal_gap_m": float(closest_gap),
        "lateral_gap_m": float(closest_lateral),
        "predicted_conflict_probability": float(max_probability),
        "active": bool(max_risk > 0.05),
        "ego_route_distance_m": float(ego_distance),
    }

