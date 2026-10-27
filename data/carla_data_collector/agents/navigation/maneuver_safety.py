"""Gap and probabilistic path checks used only for bypass/recovery intent."""

import math
import numpy as np

from agents.navigation.route_recovery import project_route, sample_route
from common.obstacle_geometry import path_hits_box


def _current(prediction):
    value = getattr(prediction, "current_position", None)
    if value is not None:
        return np.asarray(value, dtype=float)
    if len(prediction.means) > 1:
        dt = max(float(prediction.times[1] - prediction.times[0]), 1e-3)
        return prediction.means[0] - (prediction.means[1] - prediction.means[0]) / dt * prediction.times[0]
    return np.asarray(prediction.means[0], dtype=float)


def _interval_probability(mean, variance, radius):
    sigma = math.sqrt(max(float(variance), 1e-6))
    scale = sigma * math.sqrt(2.)
    return max(0., .5 * (math.erf((radius - mean) / scale) -
                         math.erf((-radius - mean) / scale)))


def observed_velocity(prediction):
    value = getattr(prediction, "current_velocity", None)
    if value is not None:
        return np.asarray(value, dtype=float)
    if len(prediction.means) > 1:
        return (prediction.means[1] - prediction.means[0]) / max(
            float(prediction.times[1] - prediction.times[0]), 1e-3)
    return np.zeros(2)


def _finite_path_distances(path, positions):
    points = np.asarray(path, dtype=float)
    segments = np.diff(points, axis=0)
    squared_lengths = np.maximum(np.sum(segments * segments, axis=1), 1e-8)
    positions = np.asarray(positions, dtype=float)
    delta = positions[:, None, :] - points[None, :-1, :]
    fraction = np.clip(np.sum(delta * segments[None], axis=2) / squared_lengths[None], 0., 1.)
    nearest = points[None, :-1] + fraction[:, :, None] * segments[None]
    return np.linalg.norm(positions[:, None] - nearest, axis=2).min(axis=1)


def choose_candidate(candidates):
    """Strictly prefer safe same-direction choices; rank both sides together."""
    same_safe = [item for item in candidates if item.get("same_direction", True) and item["safe"]]
    in_lane = [item for item in same_safe if item.get('in_lane')]
    ready = [item for item in (in_lane or same_safe or candidates) if item["ready"]]
    if not ready:
        return None
    return min(ready, key=lambda item: (
        not item.get("same_direction", True), item.get("cost", float("inf")),
        -(item.get("min_gap_m") or 0.), item["side"]))


def waiting_annotation(safety, current_state, current_reason, real_conflict=False):
    """Admission confirmation is maneuver preparation, not a predicted collision.

    Keep this independent of the speed/path hold: label fixes must not admit an
    unconfirmed lateral move or weaken an actual actor conflict.
    """
    # Once a path has been selected, its current check is authoritative.
    # Candidate diagnostics can be older and must not override a revoked path.
    confirming = (bool(safety['safe']) if 'safe' in safety else any(
        candidate.get("safe") for candidate in safety.get("candidates", [])))
    actor_blocked = safety.get('blocking_actors') or ('safe' not in safety and not confirming and any(
        candidate.get('blocking_actors') for candidate in safety.get('candidates', [])))
    if real_conflict or actor_blocked:
        return "YIELD", "predicted_conflict"
    if confirming and current_state.startswith(("AVOID", "RECOVER")):
        return current_state, current_reason
    # Geometry/sight rejection and a not-yet-selected safe candidate are not
    # evidence that a vehicle will collide with ego.
    return "YIELD", "comfortable_braking"


def _extended_projection(points, position):
    arc, lateral, tangent = project_route(points, position)
    nearest, _ = sample_route(points, arc)
    residual = float(np.dot(np.asarray(position) - nearest, tangent))
    # Local route windows truncate already-passed road. Preserve an actor's
    # real longitudinal distance instead of clamping a distant rear car to
    # the first waypoint and inventing a small receiving-lane gap.
    return arc + residual, lateral, tangent


def assess_maneuver(centerline, path, predictions, ego_position, ego_speed,
                    ego_extent, target_offset, horizon_s=3., minimum_gap_m=3.,
                    rear_ttc_s=4., probability_threshold=.2, lateral_margin_m=.35,
                    receiving_width_m=3.5, current_half_width_m=None,
                    reservation_s=0., oncoming_speed_mps=15., opposing_lane=True,
                    physical_probability=False):
    """Check the receiving lane, the swept path, and weighted future modes.

    A fast rear actor matters here because ego intends to enter its lane.
    This function never changes ordinary lane-following risk thresholds.
    """
    ego_arc = project_route(centerline, ego_position)[0]
    nominal_speed = max(float(ego_speed), 2.5)
    ego_lateral = project_route(centerline, ego_position)[1]
    # A nearly centered ego already owns the receiving lane. Braking for a
    # following actor's hypothetical catch-up cannot improve its merge safety.
    current_half_width = ego_extent[1] if current_half_width_m is None else current_half_width_m
    inside_receiving_lane = (abs(target_offset - ego_lateral) + float(current_half_width) <=
                             float(receiving_width_m) * .5 - .1)
    blockers = []
    settled_followers = []
    max_probability = 0.
    minimum_observed_gap = float("inf")
    future_samples = {}
    path_lateral_offsets = None
    for prediction in predictions:
        if not len(prediction.means):
            continue
        if prediction.actor_type == 'static':
            if path_hits_box(path, prediction.static_box, ego_extent, lateral_margin_m):
                blockers.append({'actor_id': prediction.actor_id, 'kind': 'static_on_path'})
            continue
        current = _current(prediction)
        projection = _extended_projection(centerline, current)
        if projection is None:
            continue
        arc, lateral, actor_tangent = projection
        velocity = observed_velocity(prediction)
        along_speed = float(np.dot(velocity, actor_tangent))
        lateral_speed = abs(float(np.dot(velocity, [actor_tangent[1], -actor_tangent[0]])))
        half_length = float(ego_extent[0]) + float(prediction.extent_xy[0])
        # The path tangent already supplies future orientation. Projecting
        # today's yaw into these local extents a second time over-inflates
        # the swept corridor and can deadlock a car safely moving away.
        half_width = float(ego_extent[1]) + float(prediction.extent_xy[1]) + lateral_margin_m
        delta = arc - ego_arc
        # Reserve the entire time needed to leave a rear follower's lane,
        # including the tail beyond the short probabilistic forecast. This
        # applies to paths that keep moving away and never return to that lane.
        if (prediction.actor_type == 'vehicle' and delta < -half_length and
                along_speed > float(ego_speed) + .25 and
                abs(lateral - ego_lateral) < half_width):
            if path_lateral_offsets is None:
                path_lateral_offsets = np.asarray([project_route(centerline, point)[1] for point in path])
            separation = np.abs(path_lateral_offsets - lateral)
            moving_away = (separation[-1] > abs(ego_lateral - lateral) + .5 and
                           float(separation.min()) >= abs(ego_lateral - lateral) - .1)
            if moving_away:
                tail_separation = np.minimum.accumulate(separation[::-1])[::-1]
                clear = np.flatnonzero(tail_separation >= half_width + .1)
                if len(clear):
                    distances = np.r_[0., np.cumsum(np.linalg.norm(np.diff(np.asarray(path), axis=0), axis=1))]
                    clearance_s = float(distances[clear[0]]) / nominal_speed + .5
                    catch_s = max(-delta - half_length, 0.) / max(along_speed - float(ego_speed), .01)
                    if catch_s < clearance_s:
                        blockers.append({'actor_id': prediction.actor_id, 'kind': 'source_lane_clearance',
                                         'catch_time_s': float(catch_s), 'required_s': clearance_s})
                        continue
        body_in_receiving_corridor = abs(target_offset - ego_lateral) < (
            receiving_width_m * .5 + float(current_half_width) - .2)
        settling_ahead_of_follower = (body_in_receiving_corridor and
            delta < -half_length - minimum_gap_m and along_speed <= float(ego_speed) + 1.)
        leaving_follower_lane = False
        if (prediction.actor_type == 'vehicle' and delta < -half_length - minimum_gap_m and
                -.1 <= along_speed <= float(ego_speed) + 1. and lateral_speed < .3 and
                abs(lateral - ego_lateral) < half_width and
                abs(target_offset - lateral) > abs(ego_lateral - lateral) + .5):
            if path_lateral_offsets is None:
                path_lateral_offsets = np.asarray([project_route(centerline, point)[1] for point in path])
            separation = np.abs(path_lateral_offsets - lateral)
            # Already sharing a lane with a stable follower: leaving that lane
            # increases separation. Braking and freezing the lateral escape
            # cannot improve this follower's safety. Do not apply to paths
            # which later return into its lane, fast rear cars or inward cuts.
            leaving_follower_lane = (separation[-1] > abs(ego_lateral - lateral) + .5 and
                                      float(separation.min()) >= abs(ego_lateral - lateral) - .1)
        receiving_follower = ((inside_receiving_lane or settling_ahead_of_follower) and
                              abs(lateral - target_offset) < receiving_width_m * .5)
        if ((receiving_follower or leaving_follower_lane) and prediction.actor_type == "vehicle" and
                delta < -half_length - .5 and along_speed >= -.1 and lateral_speed < .3 and
                sum(mode.probability for mode in prediction.modes
                    if mode.name.startswith("change_")) < probability_threshold):
            settled_followers.append(prediction.actor_id)
            continue
        # Receiving-lane gaps include a closing-speed check beyond a fixed
        # rear distance. They apply only while moving laterally into that lane.
        entering = not inside_receiving_lane
        in_receiving_corridor = abs(lateral - target_offset) < half_width
        if in_receiving_corridor:
            minimum_observed_gap = min(minimum_observed_gap, abs(delta) - half_length)
        if (reservation_s > 0. and in_receiving_corridor and delta > -half_length and
                (opposing_lane or along_speed < -.5)):
            # Reserve the entire pass-and-return interval, including threats
            # beyond the short Bayesian forecast. Unknown traffic may approach
            # at the configured design speed; admission requires matching sight.
            closing = nominal_speed + max(-along_speed, oncoming_speed_mps)
            gap = delta - half_length
            if gap / max(closing, .1) < reservation_s:
                blockers.append({"actor_id": prediction.actor_id, "kind": "oncoming_reservation",
                                 "gap_m": float(gap), "ttc_s": max(float(gap / closing), 0.),
                                 "required_s": float(reservation_s)})
                continue
        if entering and in_receiving_corridor:
            gap = abs(delta) - half_length
            closing = max(along_speed - float(ego_speed), 0.) if delta < 0. else max(float(ego_speed) - along_speed, 0.)
            if (gap < minimum_gap_m or
                    (closing > .25 and gap / closing < rear_ttc_s and
                     (delta < 0. or along_speed < -.5))):
                blockers.append({"actor_id": prediction.actor_id, "kind": "receiving_gap",
                                 "gap_m": float(gap), "closing_speed_mps": float(closing),
                                 "rear_ttc_s": float(max(gap, 0.) / max(closing, .01))})
                continue
        changing_mass = sum(mode.probability for mode in prediction.modes
                            if mode.name.startswith("change_"))
        if (prediction.actor_type == "vehicle" and prediction.speed_mps >= .45 and
                lateral_speed < .3 and not in_receiving_corridor and
                changing_mass < probability_threshold):
            # Diffuse covariance alone must not turn a stable parallel car
            # outside the ENTIRE proposed path into an inward-moving threat.
            # Preserve pedestrians, actual inward motion and supported cuts.
            # This applies to either travel direction. A stable oncoming car
            # in the corridor we are LEAVING is not an inward cut once its
            # observed and posterior means clear the entire finite path.
            # Endpoint tangents extend diagonally into a different lane behind
            # a recovery path. Use actual finite segments, not their infinite
            # lateral projection, for already-passed/parallel actor clearance.
            offsets = _finite_path_distances(path, [current] + list(prediction.means))
            if min(offsets) > half_width + .1:
                continue
        # A parked vehicle anywhere along the bypass/return path invalidates
        # that path even when reaching it takes longer than the forecast.
        if prediction.speed_mps < .45:
            on_path = project_route(path, current)
            if on_path is not None:
                path_arc, path_lateral, _ = on_path
                last = sample_route(path, 1e9)[0]
                end_arc = project_route(path, last)[0]
                if (abs(path_lateral) < half_width and
                        -half_length < path_arc < end_arc + half_length and
                        np.linalg.norm(current - np.asarray(ego_position)) < end_arc + half_length):
                    # Endpoint projection must not turn a far-away object
                    # beyond the path into an apparent path collision.
                    nearest = sample_route(path, path_arc)[0]
                    if np.linalg.norm(current - nearest) < half_width:
                        blockers.append({"actor_id": prediction.actor_id, "kind": "parked_on_path"})
                        continue
            # A stationary actor's observed footprint is checked along the
            # entire path above. Diffuse acceleration covariance must not turn
            # a safely bypassed parked car into a fictitious lateral mover.
            moving_intent = sum(mode.probability for mode in prediction.modes
                                if mode.name in ("accelerate", "change_left", "change_right"))
            if moving_intent < probability_threshold:
                continue
        base_max = 0.
        mode_probabilities = []
        hypotheses = [(1., prediction.means, prediction.covariances)]
        hypotheses.extend((mode.probability, mode.means, mode.covariances)
                          for mode in prediction.modes)
        for hypothesis_index, (weight, means, covariances) in enumerate(hypotheses):
            peak = 0.
            for time_s, mean, covariance in zip(prediction.times, means, covariances):
                if float(time_s) > horizon_s:
                    break
                time_key = float(time_s)
                if time_key not in future_samples:
                    future_samples[time_key] = sample_route(path, nominal_speed * time_key)
                ego_future, tangent = future_samples[time_key]
                normal = np.asarray([tangent[1], -tangent[0]])
                difference = np.asarray(mean) - ego_future
                longitudinal = float(np.dot(difference, tangent))
                lateral_distance = float(np.dot(difference, normal))
                longitudinal_variance = float(tangent.dot(covariance).dot(tangent))
                lateral_variance = float(normal.dot(covariance).dot(normal))
                probability = (_interval_probability(longitudinal, longitudinal_variance,
                                                      half_length if physical_probability else half_length + minimum_gap_m) *
                               _interval_probability(lateral_distance, lateral_variance,
                                   half_width - lateral_margin_m if physical_probability else half_width))
                peak = max(peak, probability)
            if hypothesis_index == 0:
                base_max = peak
            else:
                mode_probabilities.append(float(weight) * peak)
        actor_probability = min(1., max(base_max, sum(mode_probabilities)))
        max_probability = max(max_probability, actor_probability)
        if actor_probability >= probability_threshold:
            blockers.append({"actor_id": prediction.actor_id, "kind": "forecast_conflict",
                             "probability": float(actor_probability)})
    return {"safe": not blockers, "blocking_actors": blockers,
            "settled_follower_ids": settled_followers,
            "max_conflict_probability": float(max_probability),
            "min_gap_m": float(minimum_observed_gap) if math.isfinite(minimum_observed_gap) else None,
            "reservation_s": float(reservation_s),
            "target_offset_m": float(target_offset), "horizon_s": float(horizon_s)}


def parallel_hold_path(centerline, ego_position, lookahead_m=8., horizon_m=40.):
    """Keep the current lateral position while waiting for a receiving gap."""
    arc, lateral, _ = project_route(centerline, ego_position)
    points = [list(ego_position)]
    for distance in np.arange(2., horizon_m + .1, 2.):
        center, tangent = sample_route(centerline, arc + distance)
        points.append((center + lateral * np.asarray([tangent[1], -tangent[0]])).tolist())
    target = sample_route(points, lookahead_m)[0].tolist()
    return points, target
