"""Freeze defensive sources; record new cases, controls and targeted regressions."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from run_moving_stage3 import source_hashes
from check_maneuver_records import audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    assert not list((output/'records').glob('*.json')), 'Use a fresh output directory'
    output.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes()
    (output/'source_manifest.json').write_text(json.dumps(dict(
        started_utc=datetime.datetime.utcnow().isoformat()+'Z', source_sha256=hashes), indent=2), encoding='utf8')
    for name in hashes:
        target = output/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)

    def run(script, directory=output, extra=()):
        subprocess.run([sys.executable, str(HERE/script), '--output', str(directory)]+list(extra),
                       cwd=str(ROOT), check=True)

    run('defensive_scenarios.py')
    run('defensive_scenarios.py', output/'controls', ['--control', '--cases',
        'lead_brakes', 'cut_in_left', 'cut_in_right', 'turn_left', 'turn_right'])
    run('run_moving_maneuver_matrix.py', extra=['--cases', 'oncoming_left_dense12',
        'oncoming_right_dense24', 'pedestrian_sudden_from_left', 'pedestrian_sudden_from_right',
        'pedestrian_two_sides_from_left_dense24', 'pedestrian_two_sides_from_right_dense12'])
    run('run_static_maneuver_matrix.py', extra=['--cases', 'static_barrier_left_dense12',
        'static_barrier_right_dense24', 'static_box_in_lane_left', 'static_both_sides_blocked'])
    run('run_dense_maneuver_matrix.py', extra=['--cases', 'dense12_avoid_left_fast_rear',
        'dense24_avoid_right_fast_rear', 'dense12_recover_left_parallel',
        'dense24_early_recover_left', 'dense24_early_recover_right'])
    records = sorted((output/'records').glob('*.json'))
    rows = []
    for path in records:
        row = audit(path)
        record = json.loads(path.read_text(encoding='utf8'))
        plans = [frame['expert'].get('defensive_plan') or {} for frame in record['scene_frames']]
        name = path.stem
        constrained_stop = name == 'lead_brakes' or name.startswith('cut_in_') or name.startswith('pedestrian_sudden_')
        row['maximum_strong_deceleration_episodes'] = 1 if constrained_stop else 0
        row['minimum_required_gap_m'] = .5 if 'box_in_lane' in name else .6
        row['defensive_active_frames'] = sum(bool(plan.get('active')) for plan in plans)
        row['defensive_applied_frames'] = sum(bool(plan.get('applied')) for plan in plans)
        row['cut_in_evidence_frames'] = sum(any(c['kind'] == 'cut_in_reserve' for c in plan.get('candidates', [])) for plan in plans)
        row['turn_reserve_frames'] = sum(any(c['kind'] == 'turn_reserve' for c in plan.get('road_constraints', [])) for plan in plans)
        row['occluded_junction_frames'] = sum(bool(plan.get('junction_visibility', {}).get('occluded')) for plan in plans)
        row['visible_junction_frames'] = sum(bool(plan.get('junction_visibility', {}).get('checked')) and
            not plan['junction_visibility']['occluded'] for plan in plans)
        row['road_applied_frames'] = sum(bool(plan.get('road_applied')) for plan in plans)
        terminal_only = not row['strong_deceleration_frames'] or row['max_strong_deceleration_entry_speed_mps'] <= 2.5
        row['passed'] = bool(row['passed'] and row['road_checked'] and
            row['initial_navigation_contract_frames'] == row['frames'] and
            row['strong_deceleration_episodes'] <= row['maximum_strong_deceleration_episodes'] and terminal_only and
            row['emergency_episodes'] == 0 and (row['min_body_gap_m'] is None or row['min_body_gap_m'] >= row['minimum_required_gap_m']))
        if name in ('parallel_left', 'parallel_right', 'rear_follower', 'open_road'):
            row['passed'] = row['passed'] and row['defensive_active_frames'] == 0
        if name.startswith('cut_in_'):
            row['passed'] = row['passed'] and row['cut_in_evidence_frames'] > 0 and row['defensive_applied_frames'] > 0
        if name == 'lead_brakes':
            row['passed'] = row['passed'] and row['defensive_applied_frames'] > 0
        if name.startswith('turn_'):
            row['passed'] = row['passed'] and row['turn_reserve_frames'] > 0 and row['road_applied_frames'] > 0
        if name.endswith('_occluded'):
            row['passed'] = row['passed'] and row['occluded_junction_frames'] > 0
        rows.append(row)
        print(name, 'passed',row['passed'],'gap',row['min_body_gap_m'],'strong',row['strong_deceleration_episodes'],
              'entry',row['max_strong_deceleration_entry_speed_mps'], flush=True)
    (output/'validation.json').write_text(json.dumps(rows, indent=2), encoding='utf8')
    assert source_hashes() == hashes, 'Sources changed during simulation'
    assert len(rows) == 27 and all(row['passed'] for row in rows), 'Failed review gates: see validation.json'
    python = str(ROOT/'.venv/Scripts/python.exe')
    subprocess.run([python, str(HERE/'pack_review_scenes.py')]+[str(path) for path in records]+
        ['--output',str(output),'--audit',str(output/'validation.json')], cwd=str(ROOT), check=True, stdout=subprocess.DEVNULL)
    subprocess.run([python,str(HERE/'render_review_previews.py'),'--output',str(output)],cwd=str(ROOT),check=True)
    controls = sorted((output/'controls/records').glob('*.json'))
    control_rows = [audit(path) for path in controls]
    (output/'controls/validation.json').write_text(json.dumps(control_rows,indent=2),encoding='utf8')
    subprocess.run([python,str(HERE/'pack_review_scenes.py')]+[str(path) for path in controls]+
        ['--output',str(output/'controls'),'--audit',str(output/'controls/validation.json')],cwd=str(ROOT),check=True,stdout=subprocess.DEVNULL)
    subprocess.run([python,str(HERE/'render_review_previews.py'),'--output',str(output/'controls')],cwd=str(ROOT),check=True)
    assert source_hashes() == hashes
    print('defensive review simulation, audit, source and LMDB decode complete',flush=True)


if __name__ == '__main__':
    main()
