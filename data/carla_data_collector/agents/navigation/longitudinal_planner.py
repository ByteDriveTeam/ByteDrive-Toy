"""Deterministic short-horizon longitudinal planning for BehaviorAgent.

The caller supplies route-relative speed constraints. This planner changes
only the target speed sent to CARLA's local controller; lateral navigation and
the existing immediate collision checks remain with BehaviorAgent.
"""

import math


def _clamp(value, low, high):
    return min(max(value, low), high)


def plan_longitudinal_speed(ego_speed_mps, previous_accel_mps2,
                            desired_speed_mps, constraints,
                            horizon_s=3.0, step_s=0.25, preview_s=0.5,
                            max_accel_mps2=2.2, comfort_decel_mps2=3.0,
                            max_jerk_mps3=4.0):
    """Choose a jerk-limited speed trace and return its preview speed.

    ``constraints`` contains ``(distance_m, speed_mps)`` route stations. A
    backward braking envelope makes each limit visible before the ego reaches
    it. Constant, speed-tracking, and low-speed approach/brake intentions are
    simulated over the horizon; each acceleration obeys the jerk limit.
    ``desired_speed_mps`` is the current policy cap, including moving traffic.
    """
    speed = max(float(ego_speed_mps), 0.0)
    desired = max(float(desired_speed_mps), 0.0)
    accel_limit = max(float(max_accel_mps2), 0.1)
    decel_limit = max(float(comfort_decel_mps2), 0.1)
    jerk_limit = max(float(max_jerk_mps3), 0.1)
    previous_accel = _clamp(float(previous_accel_mps2),
                            -decel_limit, accel_limit)
    step = max(float(step_s), 0.01)
    horizon = max(float(horizon_s), step)
    count = max(1, int(math.ceil(horizon / step)))
    preview_index = min(count - 1, max(0, int(math.ceil(
        max(float(preview_s), step) / step)) - 1))
    limits = sorted((max(float(distance), 0.0), max(float(limit), 0.0))
                    for distance, limit in constraints
                    if math.isfinite(float(distance)) and
                    math.isfinite(float(limit)))

    def envelope(travelled):
        cap = desired
        for distance, limit in limits:
            remaining = max(distance - travelled, 0.0)
            cap = min(cap, math.sqrt(limit * limit +
                                     2.0 * decel_limit * remaining))
        return cap

    constant_intentions = [-decel_limit * fraction for fraction in
                           (1.0, 0.8, 0.6, 0.4, 0.2)]
    constant_intentions += [0.0]
    constant_intentions += [accel_limit * fraction for fraction in
                            (0.25, 0.5, 0.75, 1.0)]
    constant_intentions.append(previous_accel)
    candidates = [("constant", value, value, horizon)
                  for value in constant_intentions]
    # Tracking intentions can reach a low cap and then hold it. A constant
    # positive acceleration would overshoot the cap, so the optimizer could
    # otherwise prefer standing still indefinitely during red-light creep.
    candidates += [("track", response_s, 0.0, 0.0)
                   for response_s in (0.4, 0.7, 1.0, 1.5, 2.0)]
    # A constant acceleration cannot approach and then settle at a nearby
    # stop. Include accelerate-then-brake traces so a stopped ego can close the
    # last few meters instead of freezing before the stop line.
    if speed < 2.0 and any(limit < 0.5 for _, limit in limits):
        for switch_s in (0.5, 1.0, 1.5, 2.0):
            if switch_s >= horizon:
                continue
            for first in (0.5 * accel_limit, accel_limit):
                for second in (-0.5 * decel_limit, -decel_limit):
                    candidates.append(("constant", first, second, switch_s))
    best = None
    best_cost = float("inf")
    for kind, first_intention, second_intention, switch_s in candidates:
        predicted_speed = speed
        predicted_accel = previous_accel
        travelled = 0.0
        cost = 0.0
        trace = []
        for index in range(count):
            if kind == "track":
                intention = _clamp(
                    (envelope(travelled) - predicted_speed) /
                    first_intention, -decel_limit, accel_limit)
            else:
                intention = (first_intention if index * step < switch_s
                             else second_intention)
            old_accel = predicted_accel
            predicted_accel = _clamp(
                intention, old_accel - jerk_limit * step,
                old_accel + jerk_limit * step)
            next_speed = max(predicted_speed + predicted_accel * step, 0.0)
            travelled += (predicted_speed + next_speed) * 0.5 * step
            cap = envelope(travelled)
            excess = max(next_speed - cap, 0.0)
            deficit = max(cap - next_speed, 0.0)
            # Overspeed is much more costly than slow progress. The terminal
            # term prevents a candidate from looking good only at the start.
            weight = 1.0 + index / max(count - 1, 1)
            cost += step * weight * (24.0 * excess * excess +
                                     0.7 * deficit * deficit +
                                     0.015 * predicted_accel ** 2 +
                                     0.01 * ((predicted_accel - old_accel) /
                                             step) ** 2)
            trace.append((float((index + 1) * step),
                          float(travelled), float(next_speed),
                          float(predicted_accel), float(cap)))
            predicted_speed = next_speed
        if cost < best_cost - 1e-9:
            best_cost = cost
            best = trace

    command_speed = best[preview_index][2]
    # The target may temporarily exceed a newly lowered policy cap while the
    # jerk bound ramps braking. BehaviorAgent's hard collision checks still
    # override this plan when an immediate stop is necessary.
    return {
        "target_speed_mps": float(max(command_speed, 0.0)),
        "target_acceleration_mps2": float(best[0][3]),
        "horizon_s": float(count * step),
        "step_s": float(step),
        "preview_s": float(best[preview_index][0]),
        "samples": [{"time_s": t, "route_distance_m": x,
                     "speed_mps": v, "acceleration_mps2": a,
                     "speed_cap_mps": cap}
                    for t, x, v, a, cap in best],
        "constraints": [[float(distance), float(limit)]
                        for distance, limit in limits],
    }
