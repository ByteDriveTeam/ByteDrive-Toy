"""Privileged CARLA junction sight probes; advisory, never right-of-way proof."""
import math
import carla


class JunctionVisibility:
    def __init__(self, config=None):
        cfg = config or {}
        self.enabled = bool(cfg.get('defensive_junction_enabled', True))
        self.preview = float(cfg.get('defensive_road_preview_m', 40.))
        self.sight = float(cfg.get('defensive_junction_sight_m', 25.))
        self.speed = float(cfg.get('defensive_blind_junction_speed_mps', 2.5))
        self.interval = float(cfg.get('defensive_visibility_interval_s', .3))
        self._key, self._time, self._value = None, None, None

    def update(self, world, plan, pose, extent, timestamp):
        result = dict(checked=False, junction_id=None, distance_m=None, occluded=False,
                      speed_limit_mps=self.speed, probes=[], query_failed=False)
        if not self.enabled or not plan:
            return result
        distance, previous, chosen = 0., pose.location, None
        for waypoint, _ in plan:
            distance += previous.distance(waypoint.transform.location)
            previous = waypoint.transform.location
            if distance > self.preview:
                break
            if waypoint.is_junction:
                chosen = waypoint
                break
        if chosen is None:
            self._key = None
            return result
        junction = chosen.get_junction()
        result.update(junction_id=int(junction.id), distance_m=max(0., distance - extent.x - 2.))
        key = (junction.id, int(pose.location.x), int(pose.location.y))
        if self._key == key and self._time is not None and 0. <= timestamp-self._time < self.interval:
            cached = dict(self._value)
            cached['distance_m'] = result['distance_m']
            return cached
        forward = pose.get_forward_vector()
        source = carla.Location(x=pose.location.x + (extent.x + .2)*forward.x,
                                y=pose.location.y + (extent.x + .2)*forward.y,
                                z=pose.location.z + 1.2)
        approach = plan[0][0].transform.get_forward_vector()
        entries = {}
        for entry, _ in junction.get_waypoints(carla.LaneType.Driving):
            direction = entry.transform.get_forward_vector()
            if abs(direction.x*approach.x + direction.y*approach.y) > .75:
                continue
            entries[(entry.road_id, entry.section_id, entry.lane_id)] = entry
        if not entries:
            # No crossing approach was found, so no occlusion claim is made.
            result['unassessed'] = True
        try:
            for entry in entries.values():
                options = entry.previous(self.sight)
                if not options:
                    result['query_failed'] = True
                    continue
                target = options[0].transform.location
                end = carla.Location(x=target.x, y=target.y, z=target.z + 1.2)
                hits = world.cast_ray(source, end)
                blocked = any(source.distance(hit.location) < source.distance(end)-1. for hit in hits)
                result['probes'].append(dict(road_id=int(entry.road_id), lane_id=int(entry.lane_id),
                    target_world=[float(end.x), float(end.y), float(end.z)], occluded=bool(blocked)))
        except RuntimeError:
            result['query_failed'] = True
        result['checked'] = bool(result['probes'])
        result['occluded'] = result['query_failed'] or any(p['occluded'] for p in result['probes'])
        self._key, self._time, self._value = key, timestamp, dict(result)
        return result


def defensive_road_constraints(route_snapshot, junction, speed, config):
    """Early curvature and blind-junction limits as route-station constraints."""
    points = route_snapshot.get('points', [])
    preview = float(config.get('defensive_road_preview_m', 40.))
    lateral_accel = float(config.get('defensive_turn_lateral_accel_mps2', 1.5))
    reaction = float(config.get('defensive_response_s', 1.))
    constraints, sources = [], []
    for index in range(1, len(points)-1):
        before, middle, after = points[index-1:index+2]
        station = float(middle['arc_m'])
        if station > preview:
            break
        a, b, c = [p['location_world'] for p in (before, middle, after)]
        ux, uy, vx, vy = b[0]-a[0], b[1]-a[1], c[0]-b[0], c[1]-b[1]
        lengths = math.hypot(ux, uy)*math.hypot(vx, vy)*math.hypot(c[0]-a[0], c[1]-a[1])
        if lengths < .4:
            continue
        curvature = 2.*abs(ux*vy-uy*vx)/lengths
        if curvature < .025:
            continue
        distance = max(0., station - speed*reaction - 3.)
        limit = math.sqrt(lateral_accel/curvature)
        constraints.append((distance, limit))
        sources.append(dict(kind='turn_reserve', distance_m=distance, speed_limit_mps=limit,
                            curvature=float(curvature)))
    if junction.get('occluded') and junction.get('distance_m') is not None:
        distance = max(0., junction['distance_m'] - speed*reaction)
        limit = junction['speed_limit_mps']
        constraints.append((distance, limit))
        sources.append(dict(kind='blind_junction', distance_m=distance, speed_limit_mps=limit))
    return constraints, sources
