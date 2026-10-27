"""Construct oncoming/pedestrian hazards with mirrored dense safety controls."""
import argparse
import json
from pathlib import Path
import sys
import carla
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'data/carla_data_collector'))
from evaluation.route_recovery_scenarios import run_case, find_start


CASES = {
    'oncoming_left_dense12': dict(case='avoid_left', moving_type='oncoming', density=12),
    'oncoming_right_dense24': dict(case='avoid_right', moving_type='oncoming', density=24),
    'oncoming_both_blocked': dict(case='avoid_left', moving_type='oncoming', traffic_mode='occupied', density=12,
        oncoming_stop_station_m=20., max_steps=320),
    'oncoming_intrudes_from_left': dict(case='avoid_right', moving_type='oncoming', density=12,
        obstacle_lateral_m=3.5, block_other_side=False),
    'oncoming_intrudes_from_right': dict(case='avoid_left', moving_type='oncoming', density=24,
        obstacle_lateral_m=-3.5, block_other_side=False),
    'pedestrian_from_right': dict(case='avoid_right', moving_type='pedestrian', obstacle_type='walker.pedestrian.0001',
        moving_distance_m=17., moving_speed_mps=1., pedestrian_start_lateral_m=-4.),
    'pedestrian_from_left': dict(case='avoid_left', moving_type='pedestrian', obstacle_type='walker.pedestrian.0001',
        moving_distance_m=17., moving_speed_mps=1., pedestrian_start_lateral_m=4.),
    'pedestrian_both_blocked': dict(case='avoid_left', moving_type='pedestrian', obstacle_type='walker.pedestrian.0001',
        moving_distance_m=17., moving_speed_mps=1., pedestrian_start_lateral_m=-4., traffic_mode='occupied', density=12),
    'pedestrian_two_sides_from_right_dense12': dict(case='avoid_left', moving_type='pedestrian',
        obstacle_type='walker.pedestrian.0001', moving_distance_m=17., moving_speed_mps=1.,
        pedestrian_start_lateral_m=-4., block_other_side=False, density=12),
    'pedestrian_two_sides_from_left_dense24': dict(case='avoid_right', moving_type='pedestrian',
        obstacle_type='walker.pedestrian.0001', moving_distance_m=17., moving_speed_mps=1.,
        pedestrian_start_lateral_m=4., block_other_side=False, density=24),
    'pedestrian_reverse_right_dense12': dict(case='avoid_right', moving_type='pedestrian', obstacle_type='walker.pedestrian.0001',
        moving_distance_m=20., moving_speed_mps=.8, pedestrian_start_lateral_m=-4., reverse_after_s=4.5, density=12),
    'pedestrian_reverse_left_dense24': dict(case='avoid_left', moving_type='pedestrian', obstacle_type='walker.pedestrian.0001',
        moving_distance_m=20., moving_speed_mps=.8, pedestrian_start_lateral_m=4., reverse_after_s=4.5, density=24),
    'pedestrian_slow_from_right_dense12': dict(case='avoid_left', moving_type='pedestrian', obstacle_type='walker.pedestrian.0001',
        moving_distance_m=25., moving_speed_mps=.2, pedestrian_start_lateral_m=-1.5, density=12),
    'pedestrian_slow_from_left_dense24': dict(case='avoid_right', moving_type='pedestrian', obstacle_type='walker.pedestrian.0001',
        moving_distance_m=25., moving_speed_mps=.2, pedestrian_start_lateral_m=1.5, density=24),
    'pedestrian_sudden_from_right': dict(case='avoid_left', moving_type='pedestrian', obstacle_type='walker.pedestrian.0001',
        moving_distance_m=28., moving_speed_mps=1., pedestrian_start_lateral_m=-3., spawn_at_activation=True, activation_step=80),
    'pedestrian_sudden_from_left': dict(case='avoid_right', moving_type='pedestrian', obstacle_type='walker.pedestrian.0001',
        moving_distance_m=28., moving_speed_mps=1., pedestrian_start_lateral_m=3., spawn_at_activation=True, activation_step=80),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', choices=sorted(CASES), default=list(CASES))
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--braking-control', action='store_true')
    args = parser.parse_args()
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(30.)
    world = client.get_world()
    assert world.get_map().name.endswith('Town05_Opt')
    old, weather = world.get_settings(), world.get_weather()
    settings = world.get_settings()
    settings.synchronous_mode, settings.fixed_delta_seconds, settings.no_rendering_mode = True, .1, True
    world.apply_settings(settings)
    world.set_weather(carla.WeatherParameters.ClearNoon)
    config = yaml.safe_load((ROOT / 'config/default.yaml').read_text(encoding='utf8'))['carla_collector']['ego']['behavior_agent']
    if args.braking_control:
        config.update(moving_evasion_enabled=False, obstacle_bypass_enabled=False)
    try:
        for name in args.cases:
            print('START', name, flush=True)
            kwargs = dict(CASES[name])
            case = kwargs.pop('case')
            steps = min(kwargs.pop('max_steps', args.steps), args.steps)
            kwargs.setdefault('stop_on_collision', True)
            result = run_case(world, find_start(world.get_map()), config, case, steps, 0., **kwargs)
            result['scenario_name'] = name
            if args.braking_control:
                result['control_override'] = dict(moving_evasion_enabled=False, obstacle_bypass_enabled=False)
                name = 'brake_only_' + name
            if result['moving_type'] == 'pedestrian':
                result['expected_outcome'] = 'goal'
            target = args.output / 'records' / (name + '.json')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(result, indent=2), encoding='utf8')
            print('END', name, 'frames', result['frames'], 'collisions',len(result['collisions']),
                  'states', result['states'], 'goal',result['goal_reached'], flush=True)
    finally:
        world.apply_settings(old)
        world.set_weather(weather)


if __name__ == '__main__':
    main()
