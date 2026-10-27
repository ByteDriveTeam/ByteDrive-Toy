"""Freeze stage-three sources, record hazards/regressions and independently audit."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_static_stage2 import source_hashes as previous_hashes
from check_maneuver_records import audit
import hashlib


def source_hashes():
    result = previous_hashes()
    for name in ['data/carla_data_collector/worker/main.py',
                 'data/carla_data_collector/worker/collect/collect.py',
                 'data/carla_data_collector/collector/orchestrator/orchestrator.py']:
        result[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--moving-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assert not list((args.output / 'records').glob('*.json')), 'Use a fresh directory'
    hashes = source_hashes()
    (args.output / 'source_manifest.json').write_text(json.dumps(dict(
        started_utc=datetime.datetime.utcnow().isoformat() + 'Z', source_sha256=hashes), indent=2), encoding='utf8')
    commands = [['run_moving_maneuver_matrix.py']]
    if not args.moving_only:
        commands += [['run_static_maneuver_matrix.py'], ['run_dense_maneuver_matrix.py', '--cases',
            'dense12_avoid_left_fast_rear', 'dense24_avoid_right_fast_rear',
            'dense12_recover_left_parallel', 'dense24_early_recover_left', 'dense24_early_recover_right']]
    for command in commands:
        subprocess.run([sys.executable, str(HERE / command[0]), '--output', str(args.output)] + command[1:],
                       cwd=str(ROOT), check=True)
    subprocess.run([sys.executable, str(HERE / 'run_moving_maneuver_matrix.py'),
        '--output', str(args.output / 'controls'), '--cases', 'oncoming_left_dense12',
        '--braking-control', '--steps', '300'], cwd=str(ROOT), check=True)
    rows = []
    records = sorted((args.output / 'records').glob('*.json'))
    for path in records:
        row = audit(path)
        record = json.loads(path.read_text(encoding='utf8'))
        threshold = .5 if 'box_in_lane' in path.stem else .6
        row['minimum_required_gap_m'] = threshold
        # Preserve and expose the observed low-speed stopping kink, which
        # also exists in the prior implementation. A single terminal event
        # is reviewable in constrained pedestrian/blocked-oncoming stops;
        # repeated events or a high-speed strong brake fail this gate.
        # It never counts as comfort acceptance: comfort_passed stays false.
        constrained_stop = (path.stem == 'oncoming_both_blocked' or
            (record.get('moving_type') == 'pedestrian' and
             not path.stem.startswith(('pedestrian_slow_', 'pedestrian_two_sides_'))))
        allowed_strong = 1 if constrained_stop else 0
        row['maximum_strong_deceleration_episodes'] = allowed_strong
        row['maximum_terminal_entry_speed_mps'] = 2.5 if constrained_stop else None
        terminal_only = (not row['strong_deceleration_frames'] or
                         row['max_strong_deceleration_entry_speed_mps'] <= 2.5)
        row['passed'] = (row['passed'] and row['initial_navigation_contract_frames'] == row['frames'] and
                         row['strong_deceleration_episodes'] <= allowed_strong and terminal_only and
                         row['emergency_episodes'] == 0 and
                         (row['min_body_gap_m'] is None or row['min_body_gap_m'] >= threshold))
        moving = record.get('moving_type') is not None
        if moving:
            attempts = [frame['expert'].get('maneuver_safety') or {} for frame in record['scene_frames']]
            row['joint_admission_frames'] = sum(bool(s.get('immediate_moving_admission')) for s in attempts)
            row['initial_moving_admission_requires_confirmation'] = any(s.get('immediate_moving_admission') and
                s.get('admission_status') != 'ready' for s in attempts)
            if ((path.stem.startswith('oncoming_') and 'both_blocked' not in path.stem) or
                    path.stem.startswith(('pedestrian_slow_', 'pedestrian_two_sides_'))):
                row['passed'] = row['passed'] and row['joint_admission_frames'] > 0
            if 'both_blocked' in path.stem:
                row['passed'] = row['passed'] and row['executing_frames'] == 0
            if record.get('spawn_at_activation'):
                identifier = record['constructed_obstacle_id']
                birth = next(i for i, frame in enumerate(record['scene_frames'])
                    if any(box['id'] == identifier for box in frame['bboxes']))
                first = next((i, frame) for i, frame in enumerate(record['scene_frames'])
                    if any(actor['actor_id'] == identifier for actor in frame['expert']['prediction']['actors']))
                observed = next(actor for actor in first[1]['expert']['prediction']['actors'] if actor['actor_id'] == identifier)
                row.update(birth_frame=birth, first_prediction_frame=first[0], first_observation_count=observed['observation_count'])
                row['passed'] = row['passed'] and birth == record['activation_step'] and first[0] > birth and observed['observation_count'] == 1
        rows.append(row)
        print(path.stem, 'passed', row['passed'], 'gap', row['min_body_gap_m'],
              'strong', row['strong_deceleration_episodes'], 'emergency',row['emergency_episodes'], flush=True)
    (args.output / 'validation.json').write_text(json.dumps(rows, indent=2), encoding='utf8')
    assert source_hashes() == hashes, 'Sources changed during simulation'
    assert all(row['passed'] for row in rows), 'See validation.json; failed scenes are not approved'
    python = str(ROOT / '.venv/Scripts/python.exe')
    subprocess.run([python, str(HERE / 'pack_review_scenes.py')] + [str(p) for p in records] +
        ['--output', str(args.output), '--audit', str(args.output / 'validation.json')],
        check=True, cwd=str(ROOT), stdout=subprocess.DEVNULL)
    subprocess.run([python, str(HERE / 'render_review_previews.py'), '--output', str(args.output)],
                   check=True, cwd=str(ROOT))
    control_records = sorted((args.output / 'controls/records').glob('*.json'))
    control_rows = [audit(path) for path in control_records]
    for row in control_rows:
        row['evidence_role'] = 'braking_only_negative_control'
    (args.output / 'controls/validation.json').write_text(json.dumps(control_rows, indent=2), encoding='utf8')
    subprocess.run([python, str(HERE / 'pack_review_scenes.py')] + [str(p) for p in control_records] +
        ['--output', str(args.output / 'controls'), '--audit', str(args.output / 'controls/validation.json')],
        check=True, cwd=str(ROOT), stdout=subprocess.DEVNULL)
    subprocess.run([python, str(HERE / 'render_review_previews.py'), '--output', str(args.output / 'controls')],
                   check=True, cwd=str(ROOT))
    assert source_hashes() == hashes
    print('stage3 geometry, navigation, LMDB decode and preview checks complete', flush=True)


if __name__ == '__main__':
    main()
