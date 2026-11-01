"""Construct lead, evidenced cut-in, benign neighbor and Town05 turn cases."""
import argparse
import json
import math
from pathlib import Path
import sys
import carla
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'data/carla_data_collector'))
from evaluation.route_recovery_scenarios import run_case, find_start, lifted
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.local_planner import RoadOption
from agents.navigation.static_obstacles import ENV_SOLIDS, box_payload
from common.obstacle_geometry import path_hits_box


class Traffic:
    def __init__(self, name):
        self.name, self.actor, self.distance = name, None, 0.

    def setup(self, world, start, ego, actors):
        self.start = start
        if self.name.endswith('_occluded'):
            junction_wp = next(wp for wp, _ in self.plan if wp.is_junction)
            forward = start.transform.get_forward_vector()
            route = [[wp.transform.location.x, wp.transform.location.y] for wp, _ in self.plan]
            from agents.navigation.maneuver_safety import _finite_path_distances
            for entry, _ in junction_wp.get_junction().get_waypoints(carla.LaneType.Driving):
                direction = entry.transform.get_forward_vector()
                if abs(direction.x*forward.x+direction.y*forward.y) > .75:
                    continue
                options = entry.previous(12.)
                if not options:
                    continue
                pose = lifted(options[0])
                if _finite_path_distances(route, [[pose.location.x, pose.location.y]])[0] < 8.:
                    continue
                self.actor = world.try_spawn_actor(world.get_blueprint_library().find('vehicle.carlamotors.carlacola'), pose)
                if self.actor is not None:
                    self.actor.set_simulate_physics(False)
                    actors.append(self.actor)
                    return
            raise RuntimeError('Could not construct an off-route junction occluder')
        if self.name.startswith(('turn_', 'junction_')) or self.name == 'open_road':
            return
        self.distance = 28. if self.name == 'lead_brakes' else 20.
        pose = lifted(start.next(self.distance)[0])
        if self.name.startswith('cut_in'):
            pose.location += pose.get_right_vector() * (-3.5 if 'left' in self.name else 3.5)
        elif self.name in ('parallel_left', 'parallel_right', 'rear_follower'):
            pose = ego.get_transform()
            pose.location += (pose.get_forward_vector()*-12. if self.name == 'rear_follower' else
                              pose.get_right_vector()*(-3.5 if self.name == 'parallel_left' else 3.5))
        self.actor = world.spawn_actor(world.get_blueprint_library().find('vehicle.audi.tt'), pose)
        actors.append(self.actor)

    def update(self, step, ego):
        if self.actor is None or self.name.endswith('_occluded'):
            return
        if self.name in ('parallel_left', 'parallel_right', 'rear_follower'):
            pose = ego.get_transform()
            right, forward = pose.get_right_vector(), pose.get_forward_vector()
            lateral = -3.5 if self.name == 'parallel_left' else 3.5 if self.name == 'parallel_right' else 0.
            along = -12. if self.name == 'rear_follower' else 0.
            pose.location += right*lateral + forward*along
            velocity = ego.get_velocity()
        else:
            speed = max(2.-max(step-90, 0)*.1*1.5, 0.) if self.name == 'lead_brakes' else 2.
            self.distance += speed*.1
            waypoint = self.start.next(self.distance)[0]
            pose = lifted(waypoint)
            pose.location.z -= .2
            right, forward = pose.get_right_vector(), pose.get_forward_vector()
            lateral_speed, lateral = 0., 0.
            if self.name.startswith('cut_in'):
                sign = -1. if 'left' in self.name else 1.
                phase = min(max((step-65)/30., 0.), 1.)
                lateral = sign*3.5*(1.-phase)
                lateral_speed = -sign*3.5/3. if 65 <= step < 95 else 0.
                if 'release' in self.name and step >= 115:
                    phase = min((step-115)/30., 1.)
                    lateral = sign*3.5*phase
                    lateral_speed = sign*3.5/3. if step < 145 else 0.
                pose.location += right*lateral
            velocity = forward*speed + right*lateral_speed
        self.actor.set_transform(pose)
        self.actor.set_target_velocity(velocity)


def turn_routes(road_map, world):
    router = GlobalRoutePlanner(road_map, 2.)
    spawns = road_map.get_spawn_points()
    found = {}
    solids = [box_payload(bb) for label in ENV_SOLIDS
              for bb in world.get_level_bbs(getattr(carla.CityObjectLabel, label))]
    for origin in spawns:
        waypoint = road_map.get_waypoint(origin.location)
        if waypoint.is_junction:
            continue
        for destination in spawns:
            distance = origin.location.distance(destination.location)
            if not 35. < distance < 80.:
                continue
            plan = router.trace_route(origin.location, destination.location)
            options = [option for _, option in plan]
            if any(option in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT) for option in options):
                continue
            turns = set(options) & {RoadOption.LEFT, RoadOption.RIGHT}
            if len(turns) != 1 or len(plan) < 25:
                continue
            length, first_junction, previous = 0., None, plan[0][0].transform.location
            for point, _ in plan:
                length += previous.distance(point.transform.location)
                previous = point.transform.location
                if first_junction is None and point.is_junction:
                    first_junction = length
            if not 55. < length < 125. or first_junction is None or first_junction < 18.:
                continue
            if plan[-1][0].is_junction:
                continue
            side = 'left' if RoadOption.LEFT in turns else 'right'
            if side in found:
                continue
            route = [[wp.transform.location.x, wp.transform.location.y] for wp, _ in plan]
            z = plan[0][0].transform.location.z + .8
            x0, x1 = min(p[0] for p in route)-4., max(p[0] for p in route)+4.
            y0, y1 = min(p[1] for p in route)-4., max(p[1] for p in route)+4.
            # Isolate the defensive-turn feature on routes accepted by the
            # existing static-obstacle representation. Some Town05 gantry
            # boxes conservatively span the road down to ground level; those
            # routes would correctly remain blocked under that representation.
            if any(x0-math.hypot(*box['extent'][:2]) <= box['location'][0] <= x1+math.hypot(*box['extent'][:2]) and
                   y0-math.hypot(*box['extent'][:2]) <= box['location'][1] <= y1+math.hypot(*box['extent'][:2]) and
                   abs(box['location'][2]-z) <= box['extent'][2]+.8 and
                   path_hits_box(route, box, (2.4, 1.1), .35) for box in solids):
                continue
            found.setdefault(side, plan)
            if len(found) == 2:
                return found
    raise RuntimeError('Could not find both native Town05 turn routes')


CASES = ['lead_brakes', 'cut_in_left', 'cut_in_right', 'cut_in_left_release',
         'parallel_left', 'parallel_right', 'rear_follower', 'open_road', 'turn_left', 'turn_right',
         'turn_left_occluded', 'turn_right_occluded']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', choices=CASES, default=CASES)
    parser.add_argument('--control', action='store_true')
    parser.add_argument('--steps', type=int, default=650)
    args = parser.parse_args()
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(30.)
    world = client.get_world()
    if not world.get_map().name.endswith('Town05_Opt'):
        world = client.load_world('Town05_Opt')
    old, weather = world.get_settings(), world.get_weather()
    settings = world.get_settings()
    settings.synchronous_mode, settings.fixed_delta_seconds, settings.no_rendering_mode = True, .1, True
    world.apply_settings(settings)
    world.set_weather(carla.WeatherParameters.ClearNoon)
    cfg = yaml.safe_load((ROOT/'config/default.yaml').read_text(encoding='utf8'))['carla_collector']['ego']['behavior_agent']
    cfg.update(obstacle_bypass_enabled=False)
    if args.control:
        cfg['defensive_enabled'] = False
    try:
        routes = turn_routes(world.get_map(), world) if any(name.startswith('turn_') for name in args.cases) else {}
        for light in world.get_actors().filter('*traffic_light*'):
            light.set_state(carla.TrafficLightState.Green)
            light.freeze(True)
        for name in args.cases:
            print('START', name, 'control', args.control, flush=True)
            plan = routes.get(name.replace('turn_', '').replace('_occluded', ''))
            start = plan[0][0] if plan else find_start(world.get_map())
            steps = min(args.steps, 230) if name == 'lead_brakes' else args.steps
            driver = Traffic(name)
            driver.plan = plan
            record = run_case(world, start, cfg, 'recover_left', steps, 0.,
                              scenario_driver=driver, assigned_plan=plan, explicit_plan=not bool(plan),
                              stop_on_collision=True)
            record['expected_outcome'] = 'wait' if name == 'lead_brakes' else 'goal'
            record['scenario_name'] = name
            record['control_override'] = dict(obstacle_bypass_enabled=False, defensive_enabled=not args.control)
            path = args.output/'records'/((('control_' if args.control else '') + name)+'.json')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=2), encoding='utf8')
            print('END', name, record['frames'], 'collisions',len(record['collisions']), 'goal',record['goal_reached'], flush=True)
    finally:
        for light in world.get_actors().filter('*traffic_light*'):
            light.freeze(False)
        world.apply_settings(old)
        world.set_weather(weather)


if __name__ == '__main__':
    main()
