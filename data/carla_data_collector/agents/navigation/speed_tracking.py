"""Low-speed tracking without repeated throttle/brake reversals."""

import math


def terminal_front_clear(front, speed, min_gap=3., coast_decel_mps2=.25):
    """Geometric precondition only; an observed front car also needs a fresh forecast check."""
    if front.get('emergency'):
        return False
    if front.get('kind') is None:
        return True
    return bool(front.get('kind') == 'following' and front.get('speed_cap_mps', 0.) >= speed and
                front.get('gap_m', 0.) >= min_gap + 2. + speed*speed/(2.*coast_decel_mps2))


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


class LaunchAssist:
    """Bounded integral compensation after a sustained, requested stalled launch.

    Compensation needs positive controller demand and speed error. Stops and
    braking clear it immediately; ordinary launches keep the original control.
    """
    def __init__(self, delay_s=2., maximum=.45, integral_gain=.08):
        self.delay_s = float(delay_s)
        self.maximum = float(maximum)
        self.integral_gain = float(integral_gain)
        self.stalled_s = 0.
        self.active = False
        self.bias = 0.

    def limit(self, requested, previous, dt, speed, allowed=True,
              rate_per_s=.6, low_speed_mps=3., launch_throttle_max=.15,
              target_speed=None):
        elapsed = max(0., min(float(dt), .5))
        error = float(target_speed) - speed if target_speed is not None else 1.
        if not allowed or requested <= 0. or error <= .15:
            self.stalled_s = 0.
            self.active = False
            self.bias = 0.
        elif not self.active:
            if speed < .15 and requested >= .9 * launch_throttle_max:
                self.stalled_s += elapsed
                self.active = self.stalled_s >= self.delay_s
            else:
                self.stalled_s = 0.
        ceiling = self.maximum if self.active else launch_throttle_max
        if self.active and target_speed is not None:
            self.bias = min(max(self.maximum-launch_throttle_max, 0.),
                self.bias + self.integral_gain*min(error, 1.)*elapsed)
            requested = min(requested + self.bias, self.maximum,
                max(previous, 0.) + rate_per_s*elapsed)
        return limit_launch_throttle(requested, previous, dt, speed,
            rate_per_s=rate_per_s, low_speed_mps=low_speed_mps,
            launch_throttle_max=ceiling)
