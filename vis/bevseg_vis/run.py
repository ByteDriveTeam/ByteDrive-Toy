"""BEVSeg 真实 LMDB 可视化 CLI。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import load_config
from data.bevseg_synthesis import BevSegRasterizer
from vis.bevseg_vis.bevseg_vis import save_bevseg_history
from vis.data_vis.reader import SceneReader, list_scenes


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else _REPO_ROOT / path


def _pick_scene(value, root):
    scenes = list_scenes(root)
    if not scenes:
        raise FileNotFoundError("没有找到真实场景目录: {}".format(root))
    if value is None:
        return scenes[0]
    candidate = _resolve(value)
    if candidate.is_dir():
        return candidate
    by_name = root / str(value)
    if by_name.is_dir():
        return by_name
    if str(value).isdigit() and 0 <= int(value) < len(scenes):
        return scenes[int(value)]
    raise ValueError("无法解析场景: {}".format(value))


def main(argv=None):
    parser = argparse.ArgumentParser(description="从真实 CARLA LMDB 生成 BEVSeg 可视化")
    parser.add_argument("--config", default=None)
    parser.add_argument("--env", default=None)
    parser.add_argument("--scene", default=None, help="scene_XXXXXX、目录路径或场景序号")
    parser.add_argument("--frame", type=int, default=None, help="当前帧号，默认取最后一帧")
    parser.add_argument("--output", default="vis/output/bevseg.png")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.env)
    data_cfg = cfg.data.bevseg
    scene_root = _resolve(data_cfg.scene_root)
    scene_dir = _pick_scene(args.scene, scene_root)
    reader = SceneReader(scene_dir)
    try:
        if reader.failed or reader.num_frames < data_cfg.history_frames:
            raise ValueError("场景不可用或不足五帧: {}".format(scene_dir))
        frame = reader.num_frames - 1 if args.frame is None else args.frame
        if frame < data_cfg.history_frames - 1 or frame >= reader.num_frames:
            raise ValueError("frame 必须位于 [{}, {})".format(
                data_cfg.history_frames - 1, reader.num_frames))
        current_pose = reader.frame_meta(frame)["ego"]["transform"]
        rasterizer = BevSegRasterizer(data_cfg)
        map_obj = rasterizer.load_map(reader.meta)
        first = frame - data_cfg.history_frames + 1
        history = [rasterizer.rasterize_frame(
            reader.meta, reader.frame_meta(index), map_obj, current_pose)
            for index in range(first, frame + 1)]
        output = _resolve(args.output)
        save_bevseg_history(history, rasterizer.layers, output,
                            frame_ids=range(first, frame + 1))
        print("[bevseg-vis] scene={} frame={} output={}".format(scene_dir.name, frame, output))
    finally:
        reader.close()


if __name__ == "__main__":
    main()
