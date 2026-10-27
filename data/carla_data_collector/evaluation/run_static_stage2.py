"""Freeze stage-two code, simulate, audit, pack and decode review scenes."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from check_maneuver_records import audit


def source_hashes():
    paths = list((ROOT / 'data/carla_data_collector/agents/navigation').rglob('*.py'))
    paths += list((ROOT / 'data/carla_data_collector/common').glob('*.py'))
    paths += list(HERE.glob('*.py'))
    paths += [ROOT / name for name in ['config/default.yaml', 'config/schema.py',
        'data/carla_data_collector/worker/annotations/annotations.py',
        'data/carla_data_collector/worker/model_runtime/model_runtime.py',
        'data/driving_cache/driving_cache.py', 'data/driving_cache/trajectory_risk.py',
        'data/navigation_goal_filter.py', 'data/trajectory_vocabulary/trajectory_vocabulary.py']]
    paths += [ROOT / 'vis/data_vis/draw/draw.py']
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes()
    (args.output / 'source_manifest.json').write_text(json.dumps({
        'started_utc': datetime.datetime.utcnow().isoformat() + 'Z',
        'source_sha256': hashes}, indent=2), encoding='utf-8')
    subprocess.run([sys.executable, str(HERE / 'run_static_maneuver_matrix.py'),
                    '--output', str(args.output)], cwd=str(ROOT), check=True)
    subprocess.run([sys.executable, str(HERE / 'run_dense_maneuver_matrix.py'),
        '--output', str(args.output), '--cases', 'dense12_avoid_left_fast_rear',
        'dense24_avoid_right_fast_rear', 'dense12_recover_left_parallel',
        'dense24_early_recover_left', 'dense24_early_recover_right'], cwd=str(ROOT), check=True)
    records = sorted((args.output / 'records').glob('*.json'))
    assert len(records) == 13, 'Use a fresh directory for this 13-case run'
    rows = []
    for path in records:
        row = audit(path)
        in_lane = 'box_in_lane' in path.stem
        row['minimum_required_gap_m'] = .5 if in_lane else .7
        row['maximum_strong_deceleration_episodes'] = 0
        row['passed'] = (row['passed'] and row['strong_deceleration_episodes'] == 0 and
            (row['min_body_gap_m'] is None or row['min_body_gap_m'] >= row['minimum_required_gap_m']))
        record = json.loads(path.read_text(encoding='utf-8'))
        confirmations = [frame for frame in record['scene_frames']
            if (frame['expert'].get('maneuver_safety') or {}).get('admission_status') == 'confirming'
            and frame['expert']['state'].startswith('RECOVER')]
        assert all(frame['expert']['reason'].startswith('route_recover') for frame in confirmations)
        row['recovery_confirmation_frames'] = len(confirmations)
        row['false_conflict_recovery_confirmation_frames'] = 0
        if in_lane:
            active = [frame['expert']['maneuver_safety'] for frame in record['scene_frames']
                if frame['expert']['state'].startswith('AVOID')]
            row['in_lane_only'] = bool(active) and all(item.get('in_lane') for item in active)
            row['passed'] = row['passed'] and row['in_lane_only']
        if path.stem in ('static_overhead_ignored', 'static_native_environment_baseline'):
            row['passed'] = row['passed'] and row['executing_frames'] == 0
        if path.stem.startswith('static_barrier') or in_lane:
            first = next((i for i, frame in enumerate(record['scene_frames'])
                if (frame['expert'].get('maneuver_safety') or {}).get('candidates')), None)
            row['first_static_planning_frame'] = first
            row['passed'] = row['passed'] and first == 0
        rows.append(row)
        print(path.stem, 'passed', row['passed'], 'gap', row['min_body_gap_m'],
              'strong', row['strong_deceleration_episodes'], flush=True)
    (args.output / 'validation.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    assert all(row['passed'] for row in rows), 'See validation.json; failed runs are not packed'
    assert source_hashes() == hashes, 'Source changed during simulation'
    python = ROOT / '.venv/Scripts/python.exe'
    subprocess.run([str(python), str(HERE / 'pack_review_scenes.py')] +
        [str(path) for path in records] + ['--audit', str(args.output / 'validation.json'),
        '--output', str(args.output)], cwd=str(ROOT), check=True, stdout=subprocess.DEVNULL)
    subprocess.run([str(python), str(HERE / 'render_review_previews.py'), '--output', str(args.output)],
                   cwd=str(ROOT), check=True)
    print('stage2 acceptance, bbox/command decoding and previews complete', flush=True)


if __name__ == '__main__':
    main()
