"""Low-speed tracking without repeated throttle/brake reversals."""

import math


def waiting_coast_speed(distance, coast_decel_mps2=.25):
    """Reserve a coasting tail before a blocked recovery's final stop line."""
    return .65 * math.sqrt(2. * coast_decel_mps2 * max(float(distance) - 2., 0.))

def allow_coasting(speed, target_speed, policy_speed, constraints,
                   stop_for_conflict=False, wait_distance=None,
                   coast_speed_mps=3., coast_decel_mps2=.25,
                   soft_navigation_goal=False, safe_maneuver_tracking=False):
    """Coast through small tracking overshoots only with enough stopping room.

    Immediate safety stops never call this helper. Use a conservative assumed
    drag deceleration; explicit stop/curve constraints retain their braking room.
    """
    if not (.3 < speed <= coast_speed_mps and
            (soft_navigation_goal or safe_maneuver_tracking or
             (target_speed < speed and speed - target_speed <= 1.))):
        return False
    constraints = list(constraints)
    if stop_for_conflict:
        if wait_distance is None:
            return False
        constraints.append((wait_distance, 0.))
    if policy_speed <= 0. and not constraints:
        return False
    for distance, limit in constraints:
        if speed <= limit:
            continue
        stopping_room = max(speed * speed - limit * limit, 0.) / (2. * coast_decel_mps2)
        if distance < stopping_room + 2.:
            return False
    return True


def limit_launch_throttle(requested, previous, dt, speed, rate_per_s=.6,
                          low_speed_mps=3., launch_throttle_max=.15):
    if speed > low_speed_mps:
        return requested
    limited = min(requested, max(previous, 0.) + rate_per_s * max(dt, .001))
    # An initially stationary automatic gearbox can engage after several
    # ticks. Do not accumulate a large throttle demand during that delay.
    if speed < .75:
        limited = min(limited, launch_throttle_max)
    return limited
