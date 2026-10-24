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


def temporary_route(points, ego_position, side, lane_width_m,
                    obstacle_position=None, lookahead_m=8.0,
                    approach_m=18.0, clear_m=7.0, recovery_m=14.0):
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
            begin = obstacle_arc - max(float(approach_m), 4.0)
            full = obstacle_arc - 4.0
            release = obstacle_arc + max(float(clear_m), 3.0)
            finish = release + max(float(recovery_m), 4.0)
            if arc < full:
                offset = side * lane_width_m * _smooth(
                    (arc - begin) / max(full - begin, 1e-3))
            elif arc <= release:
                offset = side * lane_width_m
            else:
                offset = side * lane_width_m * (1.0 - _smooth(
                    (arc - release) / max(finish - release, 1e-3)))
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
            "phase": "avoid" if obstacle_arc is not None else "recover"}
