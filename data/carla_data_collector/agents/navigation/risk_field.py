"""Route-aligned dynamic risk field built from Bayesian actor predictions."""

import math

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


def evaluate_route_risk(route_points, predictions, ego_position, ego_speed_mps,
                        route_width_m=2.0, risk_decay_m=35.0):
    """Return scalar risk and interpretable details for a route corridor.

    ``predictions`` are :class:`ActorPrediction` objects.  The returned values
    are plain Python types so they can be written directly by msgpack.
    """
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
        corridor = float(route_width_m) + extent
        # A parallel actor whose predicted centers stay outside the route is
        # less likely to make an unobserved lane change than a generic free-space
        # Gaussian acceleration model suggests. Crossing forecasts keep full risk.
        min_mean_lateral = min(
            (_nearest_route_point(mean, points, arc, tangent)[1]
             for mean in prediction.means), default=float("inf"))
        non_crossing_prior = 0.15 if min_mean_lateral > corridor + 0.5 else 1.0
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

