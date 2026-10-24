"""Pack recorded scenario JSON into the standard scene_*/lmdb format."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data/carla_data_collector"))

from collector.writer import LmdbWriter, compact_lmdb
from vis.data_vis.reader import SceneReader


def pack(source, target, index):
    recording = json.loads(source.read_text(encoding="utf-8"))
    frames = recording["scene_frames"]
    complete = recording["distance_to_goal_m"] < 2.5
    passed = complete and not recording["collisions"]
    scene_id = "scene_{:06d}".format(index)
    directory = target / "scenes" / scene_id
    directory.mkdir(parents=True, exist_ok=True)
    meta = {
        "scene_id": scene_id, "seed": 0, "weather": "ClearNoon",
        "status": "ok" if passed else "constructed_failure", "failed": not passed,
        "failure_status": None if passed else "constructed_failure",
        "map": "Town05_Opt", "fps": 10, "sensor_dt_s": 0.1, "kinematics_dt_s": 0.1,
        "num_frames": len(frames), "num_kinematics": len(frames),
        "drive_id": recording["case"], "segment_idx": 0,
        "controller": "behavior_agent", "complete": complete, "expert_schema_version": 1,
        "route": recording["route"], "constructed_scenario": source.stem,
        "initial_body_margin_m": recording.get("initial_body_margin_m"),
        "intrinsics": {}, "extrinsics": {}, "lidar_extrinsic": None,
        "static_bboxes": recording["static_bboxes"], "traffic_lights": [],
        "camera_names": ["front"], "video_files": {},
    }
    kinematics = [{key: frame[key] for key in ("frame_id", "sim_time", "ego", "expert")}
                  for frame in frames]
    writer = LmdbWriter(directory / "lmdb", 1.0)
    try:
        writer.write_scene(meta, ({"meta": frame, "arrays": {}} for frame in frames),
                           kinematics=kinematics, est_bytes=source.stat().st_size)
    finally:
        writer.close()
    compact_lmdb(directory / "lmdb")
    reader = SceneReader(directory, open_videos=False)
    try:
        assert reader.num_frames == len(frames)
        assert len(reader.kinematics()) == len(frames)
        for i, expected in enumerate(frames):
            decoded = reader.frame_meta(i)
            assert decoded["expert"] == expected["expert"]
            assert reader.navigation_command(i, None) == expected["raw_navigation_command"]
            assert reader.recovery_command(i) == expected["recovery_command"]
        reader.frame(0)
    finally:
        reader.close()
    return {"scenario": source.stem, "scene": str(directory),
            "frames": len(frames), "collisions": len(recording["collisions"]),
            "distance_to_goal_m": recording["distance_to_goal_m"],
            "initial_body_margin_m": recording.get("initial_body_margin_m"),
            "states": recording["states"], "recovery_commands": recording["recovery_commands"],
            "passed": passed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = [pack(source, args.output, index) for index, source in enumerate(args.sources)]
    (args.output / "scenario_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
