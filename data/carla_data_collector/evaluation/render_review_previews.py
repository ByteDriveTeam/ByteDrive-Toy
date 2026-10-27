"""Render standard BEV/HUD previews without opening a GUI (repository Python)."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import cv2
from config import load_config
from vis.data_vis.reader import SceneReader
from vis.data_vis.draw.draw import render_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config()
    rows = json.loads((args.output / 'validation.json').read_text(encoding='utf-8'))
    summaries = json.loads((args.output / 'scenario_summary.json').read_text(encoding='utf-8'))
    for row, summary in zip(rows, summaries):
        scene = ROOT / summary['scene']
        reader = SceneReader(scene, open_videos=False)
        try:
            index = row['closest_frame']
            if index is None:
                index = row.get('first_execution') or 0
            state = dict(available=reader.available, show_static=False, show_rgb=True,
                         show_depth=True, show_semantic=True, show_flow=True,
                         show_bev=True, show_bbox=True, idx=index,
                         num_frames=reader.num_frames, playing=False,
                         trajectory=[item['ego']['transform'] for item in reader.kinematics()],
                         actor_trajectories=reader.actor_trajectories())
            frame = render_frame(reader.frame(index), reader.meta, cfg.data_vis, state)
            assert cv2.imwrite(str(scene / 'preview.png'), frame)
            print(scene.name, 'preview frame', index, flush=True)
        finally:
            reader.close()


if __name__ == '__main__':
    main()
