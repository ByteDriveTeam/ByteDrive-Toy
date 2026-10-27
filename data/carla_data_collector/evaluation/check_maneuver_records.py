"""Audit constructed CARLA maneuver records (collector Python + Shapely)."""

import argparse
import json
import math
import hashlib
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.navigation.maneuver_lanes import point_in_driving_corridor

_MAPS = {}


def _road_map(record):
    xml = record.get('opendrive_xml')
    if not xml:
        return None
    import carla
    key = hashlib.sha256(xml.encode('utf-8')).hexdigest()
    if key not in _MAPS:
        _MAPS[key] = carla.Map('audit', xml)
    return _MAPS[key]


def polygon(box):
    x, y = box["location"][:2]
    angle = math.radians(box["rotation"][2])
    c, s = math.cos(angle), math.sin(angle)
    ex, ey = box["extent"][:2]
    return Polygon([(x + sx * ex * c - sy * ey * s,
                     y + sx * ex * s + sy * ey * c)
                    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def audit(path):
    record = json.loads(path.read_text(encoding="utf-8"))
    closest = float("inf")
    closest_frame = None
    closest_actor = None
    contacts = []
    nearby = []
    waits = []
    executed = []
    emergencies = []
    actor_clearance = {}
    road_map = _road_map(record)
    road_violations = []
    moving_hard_brakes = []
    strong_decelerations = []
    strong_entry_speeds = []
    longitudinal_accels, lateral_accels, jerks = [], [], []
    last_accel = None
    last_time = None
    previous_velocity = None
    initial_navigation = record.get('initial_navigation')
    navigation_checks = 0
    previous_route_index = 0
    for index, frame in enumerate(record["scene_frames"]):
        expert = frame["expert"]
        safety = expert.get("maneuver_safety") or {}
        state = expert["state"]
        assert expert.get('state_id', 0) >= 0 and expert.get('reason_code', 0) >= 0, (path,index,state)
        if safety.get("waiting_for_gap"):
            waits.append(index)
            if state.startswith(("AVOID", "RECOVER")):
                assert safety.get('safe') and safety.get('admission_status') == 'confirming', (path,index,safety)
                assert expert['reason'] != 'predicted_conflict', (path,index,expert['reason'])
        if state.startswith(("AVOID", "RECOVER")):
            if not safety.get('waiting_for_gap'):
                executed.append(index)
                assert safety.get("ready"), (path, index, safety)
        assert frame["navigation_command"] == frame["raw_navigation_command"] == 0
        if initial_navigation is not None:
            commands = initial_navigation['commands']
            route_index = expert['navigation']['assigned_route_index']
            assert route_index >= previous_route_index, (path, index, 'route progress moved backwards')
            assert frame['raw_navigation_command'] == commands[min(route_index, len(commands) - 1)]
            remaining = expert['route_plan']['points']
            original_tail = initial_navigation['points'][-len(remaining):] if remaining else []
            assert [point['location_world'] for point in remaining] == original_tail
            previous_route_index = route_index
            navigation_checks += 1
        if state.startswith("AVOID"):
            assert frame["recovery_command"] == 0
        if state == "RECOVER_LEFT":
            assert frame["recovery_command"] == 2
        if state == "RECOVER_RIGHT":
            assert frame["recovery_command"] == 1
        if expert["reason"] == "emergency_stop":
            emergencies.append(index)
        ego = next(box for box in frame["bboxes"] if box["semantic"] == "ego")
        ego_polygon = polygon(ego)
        if road_map is not None:
            import carla
            for x, y in list(ego_polygon.exterior.coords)[:-1]:
                location = carla.Location(x=x, y=y, z=float(frame['ego']['transform'][2]))
                if not point_in_driving_corridor(road_map, location):
                    road_violations.append(index)
                    break
        velocity = frame['ego']['velocity']
        acceleration = frame['ego']['acceleration']
        speed = math.hypot(*velocity[:2])
        before_speed = (math.hypot(*previous_velocity[:2]) if previous_velocity is not None else
                        float(record['trace'][index]['speed_mps']))
        if max(speed, before_speed) > .5:
            if speed > .5:
                direction, direction_speed = velocity, speed
            elif previous_velocity is not None:
                direction, direction_speed = previous_velocity, before_speed
            else:
                yaw = math.radians(frame['ego']['transform'][5])
                direction, direction_speed = [math.cos(yaw), math.sin(yaw)], 1.
            longitudinal = (direction[0] * acceleration[0] + direction[1] * acceleration[1]) / direction_speed
            lateral = abs(direction[0] * acceleration[1] - direction[1] * acceleration[0]) / direction_speed
            longitudinal_accels.append(longitudinal)
            if longitudinal < -6.:
                strong_decelerations.append(index)
                strong_entry_speeds.append(before_speed)
            lateral_accels.append(lateral)
            if last_accel is not None and last_time is not None:
                jerks.append(abs(longitudinal - last_accel) / max(frame['sim_time'] - last_time, .01))
            last_accel, last_time = longitudinal, frame['sim_time']
            if frame['ego']['control']['brake'] >= .45:
                moving_hard_brakes.append(index)
        else:
            last_accel, last_time = None, None
        previous_velocity = velocity
        near = 0
        static_environment = [box for box in record.get('static_bboxes', [])
                              if box.get('semantic') in ('static', 'pole')]
        for other in list(frame["bboxes"]) + static_environment:
            if other["id"] == ego["id"]:
                continue
            distance = math.hypot(other["location"][0] - ego["location"][0],
                                  other["location"][1] - ego["location"][1])
            near += int(distance <= 60. and other.get('source') != 'env')
            if distance > 20. + math.hypot(*other['extent'][:2]) or abs(other["location"][2] - ego["location"][2]) > \
                    other["extent"][2] + ego["extent"][2]:
                continue
            identifier = (other['id'] if other['id'] is not None else
                          'env:' + hashlib.sha256(json.dumps(other, sort_keys=True).encode()).hexdigest()[:12])
            gap = ego_polygon.distance(polygon(other))
            if gap < closest:
                closest, closest_frame, closest_actor = gap, index, identifier
            actor_clearance[str(identifier)] = min(actor_clearance.get(str(identifier), float("inf")), gap)
            if gap < .001:
                contacts.append([index, identifier])
        nearby.append(near)
    expected_wait = record.get("expected_outcome") == "wait"
    goal = record.get("goal_reached", record["distance_to_goal_m"] < 2.5)
    goal_lateral, goal_heading = None, None
    aligned_goal = True
    if goal:
        final = record['scene_frames'][-1]['ego']['transform']
        endpoint = record['route']['end']
        yaw = math.radians(endpoint[5])
        goal_lateral = abs(-(final[0] - endpoint[0]) * math.sin(yaw) +
                           (final[1] - endpoint[1]) * math.cos(yaw))
        goal_heading = abs((final[5] - endpoint[5] + 180.) % 360. - 180.)
        aligned_goal = goal_lateral < .35 and goal_heading < 15. and record['distance_to_goal_m'] < 3.
    direction_case = record.get('direction_scenario')
    wrong_side = (direction_case == 'mixed_prefer_same' and any(
        frame['expert']['state'] == 'AVOID_LEFT' for frame in record['scene_frames']))
    premature_oncoming = (direction_case in ('mixed_opposing_oncoming', 'same_direction_lane_wrongway') and
                          executed and min(executed) < 100)
    passed = (not record["collisions"] and not contacts and not road_violations and
              not wrong_side and not premature_oncoming and aligned_goal and
              (not executed if expected_wait else goal))
    return {"record": str(path), "frames": record["frames"],
            "initial_navigation_contract_frames": navigation_checks,
            "initial_navigation_sha256": hashlib.sha256(json.dumps(initial_navigation,
                sort_keys=True).encode()).hexdigest() if initial_navigation else None,
            "background_vehicles": record.get("spawned_background_vehicles", 0),
            "nearby_60m_median": float(np.median(nearby)), "nearby_60m_max": max(nearby),
            "collision_events": len(record["collisions"]), "box_contact_frames": len(set(i for i, _ in contacts)),
            "min_body_gap_m": closest if math.isfinite(closest) else None,
            "closest_frame": closest_frame, "closest_actor_id": closest_actor,
            "body_gap_check_center_radius_m": 20., "body_gap_by_actor_m": actor_clearance,
            "expected_outcome": "wait" if expected_wait else "goal", "goal_reached": goal,
            "goal_alignment_passed": aligned_goal,
            "goal_lateral_error_m": goal_lateral, "goal_heading_error_deg": goal_heading,
            "waiting_frames": len(waits), "executing_frames": len(executed),
            "first_execution": min(executed) if executed else None,
            "road_checked": road_map is not None, "road_violation_frames": len(road_violations),
            "road_violations": road_violations[:20],
            "wrong_direction_selected": wrong_side, "premature_oncoming_entry": bool(premature_oncoming),
            "min_longitudinal_accel_mps2": min(longitudinal_accels) if longitudinal_accels else None,
            "max_lateral_accel_mps2": max(lateral_accels) if lateral_accels else None,
            "p95_longitudinal_jerk_mps3": float(np.percentile(jerks, 95)) if jerks else None,
            "moving_hard_brake_frames": moving_hard_brakes,
            "strong_deceleration_frames": strong_decelerations,
            "strong_deceleration_episodes": sum(i == 0 or frame != strong_decelerations[i - 1] + 1
                                                for i, frame in enumerate(strong_decelerations)),
            "max_strong_deceleration_entry_speed_mps": max(strong_entry_speeds) if strong_entry_speeds else None,
            "comfort_passed": not strong_decelerations,
            "emergency_frames": len(emergencies),
            "emergency_episodes": sum(i == 0 or frame != emergencies[i - 1] + 1
                                      for i, frame in enumerate(emergencies)),
            "passed": passed, "contacts": contacts[:20]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-clearance-m", type=float, default=.001)
    parser.add_argument("--max-strong-deceleration-episodes", type=int)
    args = parser.parse_args()
    results = [audit(path) for path in args.records]
    for row in results:
        row['minimum_required_gap_m'] = args.minimum_clearance_m
        row['passed'] = row['passed'] and (row['min_body_gap_m'] is None or
                                          row['min_body_gap_m'] >= args.minimum_clearance_m)
        row['maximum_strong_deceleration_episodes'] = args.max_strong_deceleration_episodes
        if args.max_strong_deceleration_episodes is not None:
            row['passed'] = row['passed'] and row['strong_deceleration_episodes'] <= args.max_strong_deceleration_episodes
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for row in results:
        print(Path(row["record"]).stem, "passed", row["passed"], "frames", row["frames"],
              "collisions", row["collision_events"], "contacts", row["box_contact_frames"],
              "offroad", row["road_violation_frames"] if row["road_checked"] else "unchecked",
              "min_gap", round(row["min_body_gap_m"], 3) if row["min_body_gap_m"] is not None else "no neighbor within 20m",
              "emergency episodes", row["emergency_episodes"],
              "strong deceleration episodes", row["strong_deceleration_episodes"])
    if not all(row["passed"] for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
