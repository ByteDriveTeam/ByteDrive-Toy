"""Temporary, route anchored paths for obstacle bypass and route recovery.

The assigned waypoint queue is never rewritten.  All distances are measured
along its centreline and positive lateral offsets point to its left.
"""

import math

import numpy as np


def _geometry(points):
    points = np.asarray(points, dtype=np.float64)[:, :2]
    if len(points) < 2:
        return None
    segment = np.diff(points, axis=0)
    length = np.linalg.norm(segment, axis=1)
    if np.any(length < 1e-4):
        keep = np.r_[True, length >= 1e-4]
        points = points[keep]
        segment = np.diff(points, axis=0)
        length = np.linalg.norm(segment, axis=1)
    if len(points) < 2:
        return None
    return points, segment, length, np.r_[0.0, np.cumsum(length)]


def project_route(points, position):
    """Return (arc metres, signed lateral metres, tangent) on a polyline."""
    geometry = _geometry(points)
    if geometry is None:
        return None
    route, segment, length, arc = geometry
    position = np.asarray(position, dtype=np.float64)[:2]
    fraction = np.clip(np.sum((position - route[:-1]) * segment, axis=1) /
                       (length * length), 0.0, 1.0)
    closest = route[:-1] + fraction[:, None] * segment
    index = int(np.argmin(np.sum((closest - position) ** 2, axis=1)))
    tangent = segment[index] / length[index]
    # CARLA world axes are left handed: yaw + 90 degrees points right.
    normal = np.asarray([tangent[1], -tangent[0]])
    return (float(arc[index] + fraction[index] * length[index]),
            float(np.dot(position - closest[index], normal)), tangent)


def sample_route(points, arc_m):
    geometry = _geometry(points)
    if geometry is None:
        return None
    route, segment, length, arc = geometry
    index = min(int(np.searchsorted(arc, float(arc_m), side="right")) - 1,
                len(length) - 1)
    index = max(index, 0)
    fraction = np.clip((float(arc_m) - arc[index]) / length[index], 0.0, 1.0)
    tangent = segment[index] / length[index]
    return route[index] + fraction * segment[index], tangent


def _smooth(value):
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def minimum_recovery_distance(lateral_m, ego_half_length_m, maximum_m, curvature_limit):
    """Shortest smooth return with body-length and curvature reserves.

    A cubic lateral return has maximum curvature bounded by 6*offset/length^2.
    Restarting every partial recovery at the full configured distance lets its
    endpoint retreat indefinitely after pauses and leaves the body half merged.
    """
    physical_minimum = max(4., 2. * float(ego_half_length_m) + 1.)
    curved_minimum = math.sqrt(6. * abs(float(lateral_m)) /
                               max(.8 * float(curvature_limit), 1e-3))
    return min(float(maximum_m), max(physical_minimum, curved_minimum))


def _bypass_offset(arc, side, width, obstacle_arc, approach, clear, recovery, entry):
    begin = obstacle_arc - max(float(approach), 4.)
    full = obstacle_arc - max(float(entry), 4.)
    release = obstacle_arc + max(float(clear), 3.)
    finish = release + max(float(recovery), 4.)
    if arc < full:
        return side * width * _smooth((arc - begin) / max(full - begin, 1e-3))
    if arc <= release:
        return side * width
    return side * width * (1. - _smooth((arc - release) / max(finish - release, 1e-3)))


def moving_escape_route(points, ego_position, side, offset_m, obstacle_position,
                        ego_heading, entry_m=14., clear_m=7., recovery_m=14.):
    """Enter early, hold past the moving actor's possible stopping position.

    Entry timing and return station are independent. Shaping both around one
    moving encounter point either postpones steering or returns too early.
    """
    arc, lateral, tangent = project_route(points, ego_position)
    obstacle_arc = project_route(points, obstacle_position)[0]
    end = _geometry(points)[3][-1]
    release = max(arc + entry_m + 4., obstacle_arc + clear_m)
    finish = release + recovery_m
    full = side * abs(offset_m)
    heading = np.array([math.cos(ego_heading), math.sin(ego_heading)])
    slope = float(np.dot(heading, [tangent[1], -tangent[0]])) / max(float(np.dot(heading, tangent)), .5)
    path = [list(ego_position)]
    for station in np.arange(arc + 1., min(end, finish + 2.) + .001, 1.):
        p, direction = sample_route(points, station)
        t = min((station - arc) / entry_m, 1.)
        offset = (lateral + (full - lateral) * _smooth(t) + slope * entry_m * t * (1. - t)**2
                  if station < arc + entry_m else full if station <= release else
                  full * (1. - _smooth((station - release) / recovery_m)))
        path.append((p + np.array([direction[1], -direction[0]]) * offset).tolist())
    return dict(points=path, target_world_xy=sample_route(path, 6.)[0].tolist(),
        ego_route_offset_m=float(lateral), obstacle_route_distance_m=float(obstacle_arc - arc),
        side=int(side), return_end_world_xy=sample_route(points, min(end, finish))[0].tolist(),
        return_distance_m=float(finish - arc), return_complete=bool(end >= finish), phase='avoid')


def temporary_route(points, ego_position, side, lane_width_m,
                    obstacle_position=None, lookahead_m=8.0,
                    approach_m=18.0, clear_m=7.0, recovery_m=14.0,
                    entry_clear_m=7.0, ego_heading=None):
    """Build a bypass or shortest smooth return to the assigned centreline.

    ``side`` is the bypass side.  Without an obstacle, the current signed
    offset determines the recovery direction and no alternate route is made.
    """
    projection = project_route(points, ego_position)
    if projection is None:
        return None
    ego_arc, ego_lateral, _ = projection
    obstacle_arc = None
    if obstacle_position is not None:
        projected_obstacle = project_route(points, obstacle_position)
        if projected_obstacle is None:
            return None
        obstacle_arc = projected_obstacle[0]
    end_arc = _geometry(points)[3][-1]
    required_horizon = ego_arc + 40.0
    if obstacle_arc is not None:
        required_horizon = max(
            required_horizon,
            obstacle_arc + max(float(clear_m), 3.0) +
            max(float(recovery_m), 4.0) + 2.0)
    horizon = min(required_horizon, end_arc)
    if horizon < ego_arc + 2.0:
        return None
    path = [list(np.asarray(ego_position, dtype=float)[:2])]
    initial_error, slope_error = 0., 0.
    rejoin_distance = min(max(float(recovery_m), 8.), 12.)
    if obstacle_arc is not None:
        ideal = lambda arc: _bypass_offset(arc, side, lane_width_m, obstacle_arc,
                                          approach_m, clear_m, recovery_m, entry_clear_m)
        initial_error = ego_lateral - ideal(ego_arc)
        if ego_heading is not None:
            heading_vector = np.asarray([math.cos(ego_heading), math.sin(ego_heading)])
            tangent = projection[2]
            normal = np.asarray([tangent[1], -tangent[0]])
            measured_slope = float(np.dot(heading_vector, normal)) / max(
                float(np.dot(heading_vector, tangent)), .5)
            slope_error = measured_slope - (ideal(ego_arc + .01) - ideal(ego_arc)) / .01
    step_count = max(int(math.ceil((horizon - ego_arc) / 2.0)), 2)
    for index in range(1, step_count + 1):
        arc = ego_arc + (horizon - ego_arc) * index / step_count
        sampled = sample_route(points, arc)
        if sampled is None:
            return None
        center, tangent = sampled
        normal = np.asarray([tangent[1], -tangent[0]])
        if obstacle_arc is None:
            # Start from the measured displacement; smoothly erase it over
            # the shortest comfortable distance, bounded by current speed.
            offset = ego_lateral * (1.0 - _smooth(
                (arc - ego_arc) / max(float(recovery_m), 3.0)))
        else:
            progress = min((arc - ego_arc) / rejoin_distance, 1.)
            offset = (ideal(arc) + initial_error * (1. - _smooth(progress)) +
                      slope_error * (arc - ego_arc) * (1. - progress) ** 2)
        path.append((center + normal * offset).tolist())
    target_arc = min(ego_arc + max(float(lookahead_m), 3.0), horizon)
    target_index = min(max(int(round((target_arc - ego_arc) /
                                     (horizon - ego_arc) * step_count)), 1),
                       step_count)
    return {"points": path, "target_world_xy": path[target_index],
            "ego_route_offset_m": float(ego_lateral),
            "obstacle_route_distance_m": (float(obstacle_arc - ego_arc)
                                          if obstacle_arc is not None else None),
            "side": int(side),
            "return_end_world_xy": sample_route(points, min(
                end_arc, obstacle_arc + max(float(clear_m), 3.) + max(float(recovery_m), 4.)
                if obstacle_arc is not None else ego_arc + max(float(recovery_m), 3.)))[0].tolist(),
            "return_distance_m": float((obstacle_arc + max(float(clear_m), 3.) + max(float(recovery_m), 4.)
                                         if obstacle_arc is not None else ego_arc + max(float(recovery_m), 3.)) - ego_arc),
            "return_complete": bool(obstacle_arc is None or
                                    end_arc >= obstacle_arc + max(float(clear_m), 3.) + max(float(recovery_m), 4.)),
            "phase": "avoid" if obstacle_arc is not None else "recover"}
