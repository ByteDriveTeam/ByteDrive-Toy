"""Physical lane candidates and conservative full-return geometry checks."""

import math
import numpy as np
import carla

from agents.navigation.route_recovery import project_route, sample_route


def adjacent_candidates(waypoint, allow_opposite=False):
    forward = waypoint.transform.get_forward_vector()
    normal = np.asarray([forward.y, -forward.x])
    origin = waypoint.transform.location
    results = []
    for name, adjacent, marking, permitted in (
        ("left", waypoint.get_left_lane(), waypoint.left_lane_marking,
         (carla.LaneChange.Left, carla.LaneChange.Both)),
        ("right", waypoint.get_right_lane(), waypoint.right_lane_marking,
         (carla.LaneChange.Right, carla.LaneChange.Both))):
        if adjacent is None or adjacent.lane_type != carla.LaneType.Driving or marking is None:
            continue
        direction = adjacent.transform.get_forward_vector()
        alignment = forward.x * direction.x + forward.y * direction.y
        same = alignment > .8
        if not same and (alignment > -.8 or not allow_opposite):
            continue
        if same and marking.lane_change not in permitted:
            continue
        if same:
            return_marking = adjacent.right_lane_marking if name == "left" else adjacent.left_lane_marking
            return_permitted = ((carla.LaneChange.Right, carla.LaneChange.Both) if name == "left"
                                else (carla.LaneChange.Left, carla.LaneChange.Both))
            if return_marking is None or return_marking.lane_change not in return_permitted:
                continue
        if not same:
            # CARLA's lane-change flags describe ordinary lane changes and
            # are often NONE across opposing Broken borders. Explicitly
            # enabled temporary borrowing requires Broken on BOTH sides.
            other_marking = adjacent.left_lane_marking if name == "left" else adjacent.right_lane_marking
            if (marking.type != carla.LaneMarkingType.Broken or other_marking is None or
                    other_marking.type != carla.LaneMarkingType.Broken):
                continue
        location = adjacent.transform.location
        offset = float(np.dot([location.x - origin.x, location.y - origin.y], normal))
        if abs(offset) <= 1. or abs(location.z - origin.z) > .5:
            continue
        results.append({"waypoint": adjacent, "side": 1 if offset > 0. else -1,
                        "offset_m": offset, "same_direction": same})
    return results


def point_in_driving_corridor(road_map, location, allowed=None):
    """Test the union of connected lane strips, including shared boundaries.

    CARLA can choose the wrong side of a curved shared border and then return
    None with project_to_road=False. Check both actual strips at that border.
    """
    nearest = road_map.get_waypoint(location, project_to_road=True,
                                     lane_type=carla.LaneType.Driving)
    if nearest is None:
        return False
    for lane in (nearest, nearest.get_left_lane(), nearest.get_right_lane()):
        if lane is None or lane.lane_type != carla.LaneType.Driving:
            continue
        if allowed is not None and (lane.road_id, lane.lane_id) not in allowed:
            continue
        center = lane.transform.location
        right, forward = lane.transform.get_right_vector(), lane.transform.get_forward_vector()
        dx, dy = location.x - center.x, location.y - center.y
        if (abs(dx * right.x + dy * right.y) <= lane.lane_width * .5 + .01 and
                abs(dx * forward.x + dy * forward.y) <= .5 and
                abs(location.z - center.z) < .8):
            return True
    return False


def check_path_geometry(road_map, centerline, path, origin, receiving, extent):
    if not path.get("return_complete", True):
        return {"safe": False, "kind": "incomplete_return"}
    allowed = {(origin.road_id, origin.lane_id), (receiving.road_id, receiving.lane_id)}
    maximum_curvature = 0.
    previous_heading = None
    checked = 0
    points = np.asarray(path["points"], dtype=float)
    length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    horizon = min(length, path["return_distance_m"] + 2.)
    for distance in np.arange(0., horizon + .1, 2.):
        point, tangent = sample_route(points, distance)
        if path["phase"] == "avoid":
            station, lateral_offset, _ = project_route(centerline, point)
            assigned_center = sample_route(centerline, station)[0]
            route_lane = road_map.get_waypoint(carla.Location(
                x=float(assigned_center[0]), y=float(assigned_center[1]), z=origin.transform.location.z))
            if abs(lateral_offset) + extent.y > route_lane.lane_width * .5:
                available = adjacent_candidates(route_lane, True)
                if not any((item['waypoint'].road_id, item['waypoint'].lane_id) ==
                           (receiving.road_id, receiving.lane_id) for item in available):
                    return {"safe": False, "kind": "border_not_crossable", "distance_m": float(distance)}
        heading = math.atan2(tangent[1], tangent[0])
        if previous_heading is not None:
            turn = abs((heading - previous_heading + math.pi) % (2. * math.pi) - math.pi)
            maximum_curvature = max(maximum_curvature, turn / 2.)
        previous_heading = heading
        normal = np.asarray([tangent[1], -tangent[0]])
        for longitudinal in (-extent.x, extent.x):
            for lateral in (-extent.y, extent.y):
                corner = point + longitudinal * tangent + lateral * normal
                location = carla.Location(x=float(corner[0]), y=float(corner[1]),
                                          z=origin.transform.location.z)
                if not point_in_driving_corridor(road_map, location, allowed):
                    return {"safe": False, "kind": "road_boundary", "distance_m": float(distance),
                            "corner_world": corner.tolist(),
                            "allowed": sorted(allowed)}
        checked += 1
    return {"safe": True, "checked_samples": checked,
            "max_curvature": maximum_curvature}


def check_opposing_sight(world, receiving, distance_m, ego_transform, ego_extent):
    """Require a straight visible opposing corridor through the reservation."""
    start = receiving.transform
    forward = start.get_forward_vector()
    target = receiving
    for distance in np.arange(10., distance_m + 10., 10.):
        options = receiving.previous(min(float(distance), distance_m))
        if len(options) != 1:
            return {"safe": False, "kind": "insufficient_sight_topology"}
        target = options[0]
        direction = target.transform.get_forward_vector()
        if (target.is_junction or target.road_id != receiving.road_id or
                forward.x * direction.x + forward.y * direction.y < .985 or
                abs(target.transform.location.z - start.location.z) > 1.):
            return {"safe": False, "kind": "insufficient_sight_geometry"}
    right = start.get_right_vector()
    # Observe from ego's forward side edge, not a fictitious observer already
    # standing in the lane we have yet to enter. Start just outside ego's own
    # collision mesh so the ray does not hit ego itself.
    ego_forward, ego_right = ego_transform.get_forward_vector(), ego_transform.get_right_vector()
    relative = start.location - ego_transform.location
    side = 1. if relative.x * ego_right.x + relative.y * ego_right.y > 0. else -1.
    source = carla.Location(
        x=ego_transform.location.x + ego_forward.x * (ego_extent.x + .1) +
          ego_right.x * side * max(ego_extent.y - .1, 0.),
        y=ego_transform.location.y + ego_forward.y * (ego_extent.x + .1) +
          ego_right.y * side * max(ego_extent.y - .1, 0.),
        z=ego_transform.location.z + 1.2)
    for offset in (-.7, 0., .7):
        end = carla.Location(x=target.transform.location.x + right.x * offset,
                             y=target.transform.location.y + right.y * offset,
                             z=target.transform.location.z + 1.2)
        try:
            hits = world.cast_ray(source, end)
        except RuntimeError:
            return {"safe": False, "kind": "sight_query_failed"}
        if any(source.distance(hit.location) < source.distance(end) - 1. for hit in hits):
            return {"safe": False, "kind": "occluded_sight"}
    return {"safe": True, "distance_m": float(distance_m)}
