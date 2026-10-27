"""Curvature-based tracking for admitted moving escape paths."""
import math
import numpy as np
from agents.navigation.route_recovery import project_route, sample_route


def wheelbase_from_positions(positions):
    front = (np.asarray(positions[0]) + np.asarray(positions[1])) * .5
    rear = (np.asarray(positions[2]) + np.asarray(positions[3])) * .5
    return float(np.linalg.norm((front - rear)[:2]) / 100.)


def track_committed_path(reference, position, heading_rad, blend_m=6.):
    """Join the fixed path with continuous heading, without a two-metre kink."""
    progress, _, tangent = project_route(reference, position)
    nearest = sample_route(reference, progress)[0]
    error = np.asarray(position) - nearest
    slope = np.array([math.cos(heading_rad), math.sin(heading_rad)]) - tangent
    end = project_route(reference, reference[-1])[0]
    result = [list(position)]
    for station in np.arange(progress + 1., end, 1.):
        distance = station - progress
        t = min(distance / blend_m, 1.)
        smooth = t * t * (3. - 2. * t)
        point = sample_route(reference, station)[0] + error * (1. - smooth) + slope * distance * (1. - t)**2
        result.append(point.tolist())
    result.append(reference[-1])
    return result


def pursuit_steering(path, position, heading_rad, speed_mps, wheelbase_m,
                     max_wheel_angle_rad, steering_multiplier, previous_steer):
    station = project_route(path, position)[0]
    target = sample_route(path, station + max(3.5, .35 * speed_mps + 2.)) [0]
    delta = target - np.asarray(position)
    distance_squared = max(float(np.dot(delta, delta)), .25)
    lateral_right = -math.sin(heading_rad) * delta[0] + math.cos(heading_rad) * delta[1]
    curvature = 2. * lateral_right / distance_squared
    wheel_angle = math.atan(wheelbase_m * curvature)
    steer = float(np.clip(wheel_angle / max(max_wheel_angle_rad * steering_multiplier, .1), -.8, .8))
    return float(np.clip(steer, previous_steer - .1, previous_steer + .1))
