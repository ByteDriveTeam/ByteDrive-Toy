"""Record constructed Town05 bypass/recovery into JSON for pack_review_scenes.

Run with the collector Python 3.7 environment, then pack with repository Python.
Use --recovery-offset 0.55 for pre-boundary recovery or 3.5 for wrong-lane recovery.
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import carla
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "data" / "carla_data_collector"))
from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.local_planner import RoadOption
from worker.annotations import dynamic_bboxes, static_bboxes
from worker.geometry import transform_to_list
from evaluation.scenario_traffic import ScenarioTraffic


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


def run_case(world, start, config, case, steps, recovery_offset,
             traffic_mode="none", density=0, release_step=-1, block_other_side=True,
             explicit_plan=False, obstacle_type='vehicle.audi.tt', obstacle_lateral_m=0.,
             obstacle_height_m=0., extra_static_distance_m=-1., moving_type=None,
             moving_speed_mps=2., moving_distance_m=40., activation_step=60,
             pedestrian_start_lateral_m=-4., reverse_after_s=-1.,
             oncoming_stop_station_m=None, stop_on_collision=False, spawn_at_activation=False):
    blueprint = world.get_blueprint_library()
    ego_bp = blueprint.find("vehicle.tesla.model3")
    obstacle_bp = blueprint.find(obstacle_type)
    blocker_bp = blueprint.find('vehicle.audi.tt')
    actors = []
    collisions = []
    try:
        ego = world.try_spawn_actor(ego_bp, lifted(start))
        if ego is None:
            raise RuntimeError("could not spawn ego")
        actors.append(ego)
        obstacle = None
        if case.startswith("avoid_"):
            obstacle_wp = start.next(moving_distance_m if moving_type else 25.0)[0]
            obstacle_pose = lifted(obstacle_wp)
            right = obstacle_pose.get_right_vector()
            obstacle_pose.location.x -= obstacle_lateral_m * right.x
            obstacle_pose.location.y -= obstacle_lateral_m * right.y
            obstacle_pose.location.z += obstacle_height_m
            if moving_type == 'oncoming':
                obstacle_pose.rotation.yaw += 180.
            elif moving_type == 'pedestrian':
                obstacle_pose.location.x -= pedestrian_start_lateral_m * right.x
                obstacle_pose.location.y -= pedestrian_start_lateral_m * right.y
            obstacle = None if spawn_at_activation else world.try_spawn_actor(obstacle_bp, obstacle_pose)
            if obstacle is None and not spawn_at_activation:
                raise RuntimeError("could not spawn obstacle")
            if obstacle is not None:
                obstacle.set_simulate_physics(moving_type == 'oncoming')
                actors.append(obstacle)
            if block_other_side:
                blocked_sign = (-1 if case == "avoid_left" else 1)
                blocked_wp = physical_adjacent(start.next(25.0)[0] if moving_type else obstacle_wp, blocked_sign)
                blocker = world.try_spawn_actor(blocker_bp, lifted(blocked_wp))
                if blocker is None:
                    raise RuntimeError("could not spawn side blocker")
                blocker.set_simulate_physics(False)
                actors.append(blocker)
            if extra_static_distance_m > 0.:
                extra = world.try_spawn_actor(blueprint.find('static.prop.box01'),
                                               lifted(start.next(extra_static_distance_m)[0]))
                if extra is None:
                    raise RuntimeError('could not spawn return-path static obstruction')
                extra.set_simulate_physics(False)
                actors.append(extra)
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
        if explicit_plan:
            # Generated isolated roads may lack router graph entry edges.
            # Assign the unique native lane route directly for these tests.
            agent.set_global_plan([(start, RoadOption.LANEFOLLOW)] +
                [(start.next(float(distance))[0], RoadOption.LANEFOLLOW)
                 for distance in range(2, 71, 2)])
        else:
            agent.set_destination(destination.transform.location)
        if case.startswith("recover_"):
            pose = lifted(start)
            right = pose.get_right_vector()
            sign = 1 if case == "recover_left" else -1
            pose.location.x += sign * recovery_offset * right.x
            pose.location.y += sign * recovery_offset * right.y
            ego.set_transform(pose)
            world.tick()
        traffic = ScenarioTraffic(world, start, actors, case, traffic_mode, density, release_step)
        traffic.update(0, ego)
        world.tick()
        ego.apply_control(carla.VehicleControl(brake=1.))
        # Exclude spawn free-fall from driving measurements. Keep scripted
        # traffic at its initial pose until the decision sequence begins.
        for _ in range(10):
            world.tick()
        trace = []
        scene_frames = []
        static = static_bboxes(world)
        for index in range(steps):
            traffic.update(index, ego)
            if moving_type and index >= activation_step and obstacle is not None:
                elapsed = (index - activation_step) * .1
                transform = lifted(obstacle_wp)
                forward = transform.get_forward_vector()
                right = transform.get_right_vector()
                if moving_type == 'oncoming':
                    travel = moving_speed_mps * elapsed
                    if oncoming_stop_station_m is not None:
                        travel = min(travel, moving_distance_m - oncoming_stop_station_m)
                    active_speed = moving_speed_mps if oncoming_stop_station_m is None or travel < moving_distance_m - oncoming_stop_station_m else 0.
                    offset = obstacle_lateral_m * max(1. - elapsed / 3., 0.)
                    lateral_speed = -obstacle_lateral_m / 3. if elapsed < 3. else 0.
                    transform.location.x -= travel * forward.x + offset * right.x
                    transform.location.y -= travel * forward.y + offset * right.y
                    velocity = carla.Vector3D(x=-active_speed * forward.x - lateral_speed * right.x,
                                             y=-active_speed * forward.y - lateral_speed * right.y)
                    transform.rotation.yaw = math.degrees(math.atan2(velocity.y, velocity.x)) if active_speed else transform.rotation.yaw + 180.
                else:
                    sign = -1. if pedestrian_start_lateral_m > 0. else 1.
                    walk_time = elapsed if reverse_after_s < 0. or elapsed <= reverse_after_s else 2. * reverse_after_s - elapsed
                    offset = pedestrian_start_lateral_m + sign * moving_speed_mps * walk_time
                    transform.location.x -= offset * right.x
                    transform.location.y -= offset * right.y
                    velocity = carla.Vector3D(x=-sign * moving_speed_mps * right.x, y=-sign * moving_speed_mps * right.y)
                    transform.rotation.yaw = math.degrees(math.atan2(velocity.y, velocity.x))
                obstacle.set_transform(transform)
                obstacle.set_target_velocity(velocity)
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
            if index % 100 == 0:
                safety = snapshot.get('maneuver_safety') or {}
                print('step', index, snapshot['state'], snapshot['reason'],
                      'geometry', safety.get('geometry'), flush=True)
            ego.apply_control(control)
            if spawn_at_activation and index == activation_step:
                # Birth happens after this decision and becomes observable on
                # its resulting frame; avoid exposing a just-spawned origin
                # transform as if it were the pedestrian's true location.
                obstacle = world.try_spawn_actor(obstacle_bp, obstacle_pose)
                if obstacle is None:
                    raise RuntimeError('could not spawn sudden pedestrian')
                obstacle.set_simulate_physics(False)
                actors.append(obstacle)
            world.tick()
            result = world.get_snapshot()
            snapshot['events'] = {'collision_count': len(collisions), 'lane_invasion_count': 0}
            snapshot['result_frame'] = int(result.frame)
            snapshot['result_sim_time'] = float(result.timestamp.elapsed_seconds)
            velocity, accel, angular = ego.get_velocity(), ego.get_acceleration(), ego.get_angular_velocity()
            command = agent.get_local_planner().get_navigation_command(25., 20.)
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
            destination_pose = destination.transform
            result_pose = ego.get_transform()
            lane_right = destination_pose.get_right_vector()
            lane_offset = abs((result_pose.location.x - destination_pose.location.x) * lane_right.x +
                              (result_pose.location.y - destination_pose.location.y) * lane_right.y)
            yaw_error = abs((result_pose.rotation.yaw - destination_pose.rotation.yaw + 180.) % 360. - 180.)
            goal_reached = (ego.get_location().distance(destination_pose.location) < 2.5 and
                            lane_offset < .35 and yaw_error < 15.)
            if agent.done() or goal_reached or (stop_on_collision and collisions):
                break
        states = sorted(set(item["state"] for item in trace))
        commands = sorted(set(item["navigation"].get("recovery_command")
                              for item in trace
                              if item["navigation"].get("recovery_command")))
        return {"case": case, "states": states, "recovery_commands": commands,
                "map": world.get_map().name.split('/')[-1],
                "opendrive_xml": world.get_map().to_opendrive(),
                "collisions": collisions, "frames": len(trace),
                "distance_to_goal_m": ego.get_location().distance(
                    destination.transform.location), "trace": trace,
                'scene_frames': scene_frames, 'static_bboxes': static,
                'initial_navigation': agent.get_local_planner().get_initial_navigation(25., 20.),
                'route': {'start_idx': -1, 'end_idx': -1,
                          'start': transform_to_list(start.transform),
                          'end': transform_to_list(destination.transform)},
                'initial_body_margin_m': start.lane_width*.5-ego.bounding_box.extent.y-recovery_offset
                    if case.startswith('recover_') else None,
                'traffic_mode': traffic_mode, 'background_vehicles': density,
                'spawned_background_vehicles': len(traffic.tags.get('background', [])),
                'traffic_actor_ids': traffic.tags, 'release_step': release_step,
                'goal_reached': bool(agent.done() or goal_reached),
                'constructed_obstacle_type': obstacle_type,
                'constructed_obstacle_id': obstacle.id if obstacle is not None else None,
                'constructed_obstacle_lateral_m': obstacle_lateral_m,
                'constructed_obstacle_height_m': obstacle_height_m,
                'extra_static_distance_m': extra_static_distance_m,
                'moving_type': moving_type, 'moving_speed_mps': moving_speed_mps,
                'moving_distance_m': moving_distance_m, 'activation_step': activation_step,
                'pedestrian_start_lateral_m': pedestrian_start_lateral_m,
                'reverse_after_s': reverse_after_s,
                'oncoming_stop_station_m': oncoming_stop_station_m,
                'spawn_at_activation': spawn_at_activation,
                'expected_outcome': 'wait' if traffic_mode == 'occupied' and release_step < 0 else 'goal'}
    finally:
        for actor in reversed(actors):
            try:
                if actor.type_id.startswith("sensor."):
                    actor.stop()
                actor.destroy()
            except RuntimeError:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("avoid_left", "avoid_right",
                                          "recover_left", "recover_right"),
                        required=True)
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recovery-offset", type=float, default=0.55)
    parser.add_argument("--traffic-mode", choices=("none", "occupied", "fast_rear", "parallel", "late_intrusion"), default="none")
    parser.add_argument("--background-vehicles", type=int, default=0)
    parser.add_argument("--release-step", type=int, default=-1)
    parser.add_argument("--reset-world", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--obstacle-type', default='vehicle.audi.tt')
    parser.add_argument('--obstacle-lateral-m', type=float, default=0.)
    parser.add_argument('--obstacle-height-m', type=float, default=0.)
    parser.add_argument('--extra-static-distance-m', type=float, default=-1.)
    args = parser.parse_args()
    random.seed(args.seed)
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(30.0)
    world = client.get_world()
    if args.reset_world or not world.get_map().name.endswith("Town05_Opt"):
        world = client.load_world("Town05_Opt")
    old = world.get_settings()
    old_weather = world.get_weather()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1
    settings.no_rendering_mode = True
    world.apply_settings(settings)
    world.set_weather(carla.WeatherParameters.ClearNoon)
    world.tick()
    try:
        config = yaml.safe_load((ROOT / "config" / "default.yaml").read_text(
            encoding="utf-8"))["carla_collector"]["ego"]["behavior_agent"]
        result = run_case(world, find_start(world.get_map()), config,
                          args.case, args.steps, args.recovery_offset,
                          args.traffic_mode, args.background_vehicles, args.release_step,
                          obstacle_type=args.obstacle_type, obstacle_lateral_m=args.obstacle_lateral_m,
                          obstacle_height_m=args.obstacle_height_m, extra_static_distance_m=args.extra_static_distance_m)
        result['seed'] = args.seed
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        summary = {key: value for key, value in result.items()
                   if key not in ('trace', 'scene_frames', 'static_bboxes', 'route',
                                  'collisions', 'traffic_actor_ids', 'opendrive_xml', 'initial_navigation')}
        summary['collision_events'] = len(result['collisions'])
        print(json.dumps(summary, indent=2))
    finally:
        world.apply_settings(old)
        world.set_weather(old_weather)


if __name__ == "__main__":
    main()
