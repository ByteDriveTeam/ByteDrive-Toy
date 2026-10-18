"""Route-aligned dynamic risk field built from Bayesian actor predictions."""

import math

import numpy as np


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
    return index, float(distance[index]), float(max(arc[index] + forward, 0.0))


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
        for step, (mean, covariance) in enumerate(zip(
                prediction.means, prediction.covariances)):
            index, distance, actor_arc = _nearest_route_point(mean, points, arc, tangent)
            time_s = float(prediction.times[min(step, len(prediction.times) - 1)])
            longitudinal = actor_arc - ego_arc
            if longitudinal < -2.0:
                continue
            lateral = distance
            extent = max(prediction.extent_xy[1], 0.25)
            corridor = float(route_width_m) + extent
            if lateral > corridor + 4.0:
                continue
            tangent_vec = tangent[index]
            normal = np.asarray([-tangent_vec[1], tangent_vec[0]])
            rel = np.asarray(mean[:2], dtype=np.float64) - points[index]
            lateral_signed = float(np.dot(rel, normal))
            variance = max(float(normal.dot(covariance).dot(normal)), 0.05)
            longitudinal_variance = max(float(tangent_vec.dot(covariance).dot(tangent_vec)), 0.05)
            sigma_lateral = math.sqrt(variance) + extent
            sigma_longitudinal = math.sqrt(longitudinal_variance) + max(prediction.extent_xy[0], 0.5)
            lateral_probability = math.exp(-0.5 * (lateral_signed / sigma_lateral) ** 2)
            future_gap = longitudinal - max(float(ego_speed_mps), 0.0) * time_s
            longitudinal_probability = math.exp(-0.5 * (future_gap / sigma_longitudinal) ** 2)
            probability = float(np.clip(lateral_probability * longitudinal_probability, 0.0, 1.0))
            distance_weight = math.exp(-max(longitudinal, 0.0) / max(risk_decay_m, 1e-3))
            actor_weight = 1.15 if prediction.actor_type == "pedestrian" else 1.0
            risk = float(np.clip(probability * distance_weight * actor_weight, 0.0, 1.0))
            max_risk = max(max_risk, risk)
            max_probability = max(max_probability, probability)
            if longitudinal >= 0.0 and longitudinal < closest_gap:
                closest_gap = longitudinal
                closest_lateral = abs(lateral_signed)
                closest = prediction
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

