"""Construct static obstacles and record stage-two review scenarios."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


# name, maneuver, asset, density, traffic, lateral, height, return obstruction, steps, expected
CASES = (
    ('static_barrier_left_dense12', 'avoid_left', 'static.prop.streetbarrier', 12, 'fast_rear', 0., 0., -1., 650, 'goal'),
    ('static_barrier_right_dense24', 'avoid_right', 'static.prop.streetbarrier', 24, 'fast_rear', 0., 0., -1., 700, 'goal'),
    ('static_box_in_lane_left', 'avoid_left', 'static.prop.box01', 12, 'none', -1.7, 0., -1., 600, 'goal'),
    ('static_box_in_lane_right', 'avoid_right', 'static.prop.box01', 12, 'none', 1.7, 0., -1., 600, 'goal'),
    ('static_both_sides_blocked', 'avoid_left', 'static.prop.streetbarrier', 24, 'occupied', 0., 0., -1., 220, 'wait'),
    ('static_return_path_blocked', 'avoid_left', 'static.prop.streetbarrier', 0, 'none', 0., 0., 45., 220, 'wait'),
    ('static_overhead_ignored', 'avoid_left', 'static.prop.box01', 0, 'none', 0., 6., -1., 500, 'goal'),
    ('static_native_environment_baseline', 'recover_left', 'vehicle.audi.tt', 0, 'none', 0., 0., -1., 500, 'goal'),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='*')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    recorder = Path(__file__).with_name('route_recovery_scenarios.py')
    for name, maneuver, asset, density, traffic, lateral, height, extra, steps, expected in CASES:
        if args.cases and name not in args.cases:
            continue
        path = args.output / 'records' / (name + '.json')
        print('scenario', name, flush=True)
        subprocess.run([sys.executable, str(recorder), '--case', maneuver,
            '--obstacle-type', asset, '--background-vehicles', str(density), '--traffic-mode', traffic,
            '--obstacle-lateral-m', str(lateral), '--obstacle-height-m', str(height),
            '--extra-static-distance-m', str(extra), '--recovery-offset', '0',
            '--steps', str(steps), '--seed', '0', '--reset-world', '--output', str(path)], check=True)
        record = json.loads(path.read_text(encoding='utf-8'))
        record['expected_outcome'] = expected
        record['static_scenario'] = name
        path.write_text(json.dumps(record, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
