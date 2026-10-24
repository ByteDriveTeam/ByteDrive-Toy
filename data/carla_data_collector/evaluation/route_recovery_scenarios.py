"""Record constructed Town05 bypass/recovery into JSON for pack_review_scenes.

Run with the collector Python 3.7 environment, then pack with repository Python.
Use --recovery-offset 0.55 for pre-boundary recovery or 3.5 for wrong-lane recovery.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import carla
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "data" / "carla_data_collector"))
from agents.navigation.behavior_agent import BehaviorAgent
from common.navigation_command import navigation_command_from_plan
from worker.annotations import dynamic_bboxes, static_bboxes
from worker.geometry import transform_to_list


def find_start(road_map):
    return min((point for point in road_map.generate_waypoints(2.0)
                if point.road_id == 38 and point.lane_id == -2 and
                not point.is_junction),
               key=lambda point: abs(point.transform.location.y - 100.0))


def lifted(waypoint):
    pose = waypoint.transform
    pose.location.z += 0.25
    return pose


def physical_adjacent(waypoint, sign):
    forward = waypoint.transform.get_forward_vector()
    normal = (forward.y, -forward.x)
    origin = waypoint.transform.location
    return next(lane for lane in (waypoint.get_left_lane(),
                                  waypoint.get_right_lane())
                if lane is not None and sign * (
                    (lane.transform.location.x - origin.x) * normal[0] +
                    (lane.transform.location.y - origin.y) * normal[1]) > 1.0)


def run_case(world, start, config, case, steps, recovery_offset):
    blueprint = world.get_blueprint_library()
    ego_bp = blueprint.find("vehicle.tesla.model3")
    obstacle_bp = blueprint.find("vehicle.audi.tt")
    actors = []
    collisions = []
    try:
        ego = world.try_spawn_actor(ego_bp, lifted(start))
        if ego is None:
            raise RuntimeError("could not spawn ego")
        actors.append(ego)
        obstacle = None
        if case.startswith("avoid_"):
            obstacle_wp = start.next(25.0)[0]
            obstacle = world.try_spawn_actor(obstacle_bp, lifted(obstacle_wp))
            if obstacle is None:
                raise RuntimeError("could not spawn obstacle")
            obstacle.set_simulate_physics(False)
            actors.append(obstacle)
            blocked_sign = (-1 if case == "avoid_left" else 1)
            blocked_wp = physical_adjacent(obstacle_wp, blocked_sign)
            blocker = world.try_spawn_actor(obstacle_bp, lifted(blocked_wp))
            if blocker is None:
                raise RuntimeError("could not spawn side blocker")
            blocker.set_simulate_physics(False)
            actors.append(blocker)
        sensor = world.spawn_actor(
            blueprint.find("sensor.other.collision"), carla.Transform(),
            attach_to=ego)
        sensor.listen(lambda event: collisions.append({
            "frame": int(event.frame), "other_id": int(event.other_actor.id),
            "other_type": event.other_actor.type_id}))
        actors.append(sensor)
        # CARLA reports a just-spawned actor at the origin until the first
        # synchronous tick.  The planner must see the actual spawn location.
        world.tick()
        destination = start.next(70.0)[0]
        agent = BehaviorAgent(ego, behavior="normal", behavior_config=config)
        agent.set_destination(destination.transform.location)
        if case.startswith("recover_"):
            pose = lifted(start)
            right = pose.get_right_vector()
            sign = 1 if case == "recover_left" else -1
            pose.location.x += sign * recovery_offset * right.x
            pose.location.y += sign * recovery_offset * right.y
            ego.set_transform(pose)
            world.tick()
        trace = []
        scene_frames = []
        static = static_bboxes(world)
        for index in range(steps):
            control = agent.run_step()
            snapshot = agent.get_decision_snapshot()
            pose = ego.get_transform()
            speed = ego.get_velocity()
            maneuver = snapshot.get("maneuver_plan") or {}
            trace.append({
                "step": index, "world_frame": int(world.get_snapshot().frame),
                "location_world": [pose.location.x, pose.location.y],
                "speed_mps": math.hypot(speed.x, speed.y),
                "state": snapshot["state"], "state_id": snapshot["state_id"],
                "reason": snapshot["reason"], "reason_code": snapshot["reason_code"],
                "navigation": snapshot["navigation"],
                "maneuver": {name: maneuver.get(name) for name in (
                    "phase", "side", "ego_route_offset_m",
                    "obstacle_route_distance_m", "target_world_xy",
                    "obstacle_actor_id")},
                "target_speed_mps": snapshot["target_speed_mps"],
                "control": {"steer": control.steer, "throttle": control.throttle,
                            "brake": control.brake},
            })
            ego.apply_control(control)
            world.tick()
            result = world.get_snapshot()
            snapshot['events'] = {'collision_count': len(collisions), 'lane_invasion_count': 0}
            snapshot['result_frame'] = int(result.frame)
            snapshot['result_sim_time'] = float(result.timestamp.elapsed_seconds)
            velocity, accel, angular = ego.get_velocity(), ego.get_acceleration(), ego.get_angular_velocity()
            command = navigation_command_from_plan(ego.get_location(), agent.get_local_planner().get_plan(), 25., 20.)
            scene_frames.append({
                'frame_id': int(result.frame), 'sim_time': float(result.timestamp.elapsed_seconds),
                'ego': {'transform': transform_to_list(ego.get_transform()),
                        'velocity': [velocity.x,velocity.y,velocity.z],
                        'acceleration': [accel.x,accel.y,accel.z],
                        'angular_velocity': [angular.x,angular.y,angular.z],
                        'control': {'steer': control.steer, 'throttle': control.throttle,
                                    'brake': control.brake, 'hand_brake': False, 'reverse': False}},
                'expert': snapshot, 'navigation_command': command, 'raw_navigation_command': command,
                'recovery_command': snapshot['navigation'].get('recovery_command',0),
                'bboxes': dynamic_bboxes(world,ego.id), 'traffic_light_states': [],
                'relevant_traffic_control': {},
            })
            if agent.done():
                break
        states = sorted(set(item["state"] for item in trace))
        commands = sorted(set(item["navigation"].get("recovery_command")
                              for item in trace
                              if item["navigation"].get("recovery_command")))
        return {"case": case, "states": states, "recovery_commands": commands,
                "collisions": collisions, "frames": len(trace),
                "distance_to_goal_m": ego.get_location().distance(
                    destination.transform.location), "trace": trace,
                'scene_frames': scene_frames, 'static_bboxes': static,
                'route': {'start_idx': -1, 'end_idx': -1,
                          'start': transform_to_list(start.transform),
                          'end': transform_to_list(destination.transform)},
                'initial_body_margin_m': start.lane_width*.5-ego.bounding_box.extent.y-recovery_offset
                    if case.startswith('recover_') else None}
    finally:
        for actor in reversed(actors):
            if actor.is_alive:
                if actor.type_id.startswith("sensor."):
                    actor.stop()
                actor.destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("avoid_left", "avoid_right",
                                          "recover_left", "recover_right"),
                        required=True)
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recovery-offset", type=float, default=0.55)
    args = parser.parse_args()
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(30.0)
    world = client.get_world()
    if not world.get_map().name.endswith("Town05_Opt"):
        world = client.load_world("Town05_Opt")
    old = world.get_settings()
    old_weather = world.get_weather()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1
    settings.no_rendering_mode = True
    world.apply_settings(settings)
    world.set_weather(carla.WeatherParameters.ClearNoon)
    try:
        config = yaml.safe_load((ROOT / "config" / "default.yaml").read_text(
            encoding="utf-8"))["carla_collector"]["ego"]["behavior_agent"]
        result = run_case(world, find_start(world.get_map()), config,
                          args.case, args.steps, args.recovery_offset)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in result.items()
                          if key not in ('trace','scene_frames','static_bboxes','route')}, indent=2))
    finally:
        world.apply_settings(old)
        world.set_weather(old_weather)


if __name__ == "__main__":
    main()
