"""Run stage-one CARLA acceptance, audit, pack and decode review scenes.

Use collector Python with CARLA on port 2000; no other simulator runner may run.
"""

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def source_hashes():
    sources = list((ROOT / 'data/carla_data_collector/agents/navigation').glob('*.py'))
    sources += list(HERE.glob('*.py'))
    sources += [ROOT / 'config/default.yaml', ROOT / 'config/schema.py']
    sources += [ROOT / name for name in (
        'data/navigation_goal_filter.py', 'data/driving_cache/driving_cache.py',
        'data/trajectory_vocabulary/trajectory_vocabulary.py')]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(sources)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes()
    (args.output / 'source_manifest.json').write_text(json.dumps({
        'started_utc': datetime.datetime.utcnow().isoformat() + 'Z',
        'python': sys.version, 'source_sha256': hashes}, indent=2), encoding='utf-8')
    for runner in ('direction_scenarios.py', 'run_dense_maneuver_matrix.py'):
        subprocess.run([sys.executable, str(HERE / runner), '--output', str(args.output)],
                       cwd=str(ROOT), check=True)
    records = sorted((args.output / 'records').glob('*.json'))
    assert len(records) == 15, 'Use a fresh output directory for this 15-case acceptance run'
    audit = args.output / 'validation.json'
    subprocess.run([sys.executable, str(HERE / 'check_maneuver_records.py')] +
                   [str(path) for path in records] + ['--output', str(audit),
                    '--minimum-clearance-m', '.7',
                    '--max-strong-deceleration-episodes', '0'], cwd=str(ROOT), check=True)
    assert source_hashes() == hashes, 'Control/test source changed during acceptance'
    repository_python = ROOT / '.venv/Scripts/python.exe'
    subprocess.run([str(repository_python), str(HERE / 'pack_review_scenes.py')] +
                   [str(path) for path in records] + ['--audit', str(audit),
                    '--output', str(args.output)], cwd=str(ROOT), check=True)
    subprocess.run([str(repository_python), str(HERE / 'render_review_previews.py'),
                    '--output', str(args.output)], cwd=str(ROOT), check=True)
    print('stage1 acceptance and scene decoding complete', flush=True)


if __name__ == '__main__':
    main()
