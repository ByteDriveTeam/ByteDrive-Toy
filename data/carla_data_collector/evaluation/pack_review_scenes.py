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


def pack(source, target, index, validation=None):
    recording = json.loads(source.read_text(encoding="utf-8"))
    frames = recording["scene_frames"]
    goal_reached = recording.get("goal_reached", recording["distance_to_goal_m"] < 2.5)
    expected_wait = recording.get("expected_outcome") == "wait"
    held = not any(state.startswith(("AVOID", "RECOVER")) for state in recording["states"])
    passed = not recording["collisions"] and (held if expected_wait else goal_reached)
    if validation is not None:
        assert Path(validation["record"]).resolve() == source.resolve()
        passed = passed and validation["passed"]
    complete = goal_reached or (expected_wait and passed)
    scene_id = "scene_{:06d}".format(index)
    directory = target / "scenes" / scene_id
    directory.mkdir(parents=True, exist_ok=True)
    meta = {
        "scene_id": scene_id, "seed": recording.get("seed", 0), "weather": "ClearNoon",
        "status": ("max_frames" if expected_wait else "ok") if passed else "constructed_failure", "failed": not passed,
        "failure_status": None if passed else "constructed_failure",
        "map": recording.get("map", "Town05_Opt"), "fps": 10, "sensor_dt_s": 0.1, "kinematics_dt_s": 0.1,
        "num_frames": len(frames), "num_kinematics": len(frames),
        "drive_id": recording["case"], "segment_idx": 0,
        "controller": "behavior_agent", "complete": complete, "expert_schema_version": 1,
        "route": recording["route"], "constructed_scenario": source.stem,
        "opendrive_xml": recording.get("opendrive_xml"),
        "initial_body_margin_m": recording.get("initial_body_margin_m"),
        "static_scenario": recording.get('static_scenario'),
        "constructed_obstacle_type": recording.get('constructed_obstacle_type'),
        "constructed_obstacle_id": recording.get('constructed_obstacle_id'),
        "goal_reached": goal_reached, "expected_outcome": "wait" if expected_wait else "goal",
        "traffic_mode": recording.get("traffic_mode", "none"),
        "background_vehicles": recording.get("spawned_background_vehicles", 0),
        "maneuver_validation": validation,
        "intrinsics": {}, "extrinsics": {}, "lidar_extrinsic": None,
        "static_bboxes": recording["static_bboxes"], "traffic_lights": [],
        "initial_navigation": recording.get('initial_navigation'),
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
        assert reader.meta['static_bboxes'] == recording['static_bboxes']
        assert reader.meta.get('initial_navigation') == recording.get('initial_navigation')
        for i, expected in enumerate(frames):
            decoded = reader.frame_meta(i)
            assert decoded["expert"] == expected["expert"]
            assert decoded['bboxes'] == expected['bboxes']
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
            "expected_outcome": "wait" if expected_wait else "goal",
            "goal_reached": goal_reached,
            "passed": passed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path,
                        help="Optional geometry audit; every source must have a matching result")
    args = parser.parse_args()
    validations = None
    if args.audit:
        validations = {Path(row["record"]).resolve(): row for row in
                       json.loads(args.audit.read_text(encoding="utf-8"))}
    summary = [pack(source, args.output, index,
                    validations[source.resolve()] if validations is not None else None)
               for index, source in enumerate(args.sources)]
    (args.output / "scenario_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
