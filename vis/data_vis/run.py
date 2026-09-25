"""可视化入口 CLI：浏览原始场景或渲染新驾驶数据集的训练样本。

模块: vis/data_vis/run.py
依赖: argparse, sys, pathlib, cv2, config.load_config, data.driving_dataset, vis.data_vis.reader/viewer/driving_sample
读取配置: data_vis.scene_root/driving.output_dir/tile_width_px/tile_height_px, data.driving.scene_root
对外接口:
    - main() -> None     # 解析命令行并启动可视化
说明: --dataset raw 保留原始交互窗口；--dataset driving 输出五帧输入及独立占用、场与轨迹监督面板。
      从仓库根运行：./.venv/Scripts/python.exe vis/data_vis/run.py --dataset driving --scene scene_000000
"""

import argparse
import sys
from pathlib import Path

import cv2

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import load_config
from data.driving_dataset import DrivingDataset
from vis.data_vis.driving_sample import render_driving_sample
from vis.data_vis.reader import SceneReader, list_scenes
from vis.data_vis.viewer import Viewer


def _resolve(path):
    p = Path(path)
    return p if p.is_absolute() else _REPO_ROOT / p


def _pick_scene(arg, scene_root):
    """把 --scene 解析成具体场景目录：路径 / 场景名 / 索引 / 缺省取第一个。"""
    scenes = list_scenes(scene_root)
    if arg is None:
        assert scenes, "scene_root 下无 scene_* 目录: {}".format(scene_root)
        return scenes[0]
    candidate = _resolve(arg)
    if candidate.is_dir():
        return candidate
    by_name = scene_root / str(arg)
    if by_name.is_dir():
        return by_name
    assert str(arg).isdigit(), "--scene 既非目录/场景名，也非整数索引: {}".format(arg)
    idx = int(arg)
    assert 0 <= idx < len(scenes), "场景索引 {} 越界（共 {} 个）".format(idx, len(scenes))
    return scenes[idx]


def main():
    parser = argparse.ArgumentParser(description="ByteDrive 采集数据集可视化")
    parser.add_argument("--config", default=None, help="主配置文件路径（缺省 config/default.yaml）")
    parser.add_argument("--env", default=None, help="环境覆盖名（叠加 config/<env>.yaml）")
    parser.add_argument("--scene", default=None, help="场景目录/场景名/索引（缺省取第一个）")
    parser.add_argument("--dataset", choices=("raw", "driving"), default="raw",
                        help="raw=原始传感器交互窗口；driving=新驾驶训练样本监督图")
    parser.add_argument("--frame", type=int, default=0, help="driving 模式起始帧号")
    parser.add_argument("--count", type=int, default=1, help="driving 模式输出连续帧数")
    args = parser.parse_args()

    cfg = load_config(args.config, args.env)
    scene_root = _resolve(cfg.data.driving.scene_root if args.dataset == "driving"
                          else cfg.data_vis.scene_root)
    scene_dir = _pick_scene(args.scene, scene_root)
    print("[vis] 打开场景:", scene_dir)

    if args.dataset == "driving":
        if args.frame < 0 or args.count < 1:
            parser.error("--frame 须非负，--count 须至少为 1")
        if scene_dir.parent.resolve() != scene_root.resolve():
            parser.error("driving 模式的 --scene 须位于 data.driving.scene_root 下")
        dataset = DrivingDataset(cfg, scene_name=scene_dir.name)
        output_dir = _resolve(cfg.data_vis.driving.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            selected = [(index, frame_idx) for index, (_, frame_idx) in enumerate(dataset.frame_index)
                        if args.frame <= frame_idx < args.frame + args.count]
            if not selected:
                parser.error("所选场景没有匹配 --frame/--count 的样本")
            for index, frame_idx in selected:
                canvas = render_driving_sample(dataset[index], cfg, scene_dir.name, frame_idx)
                path = output_dir / "{}_f{:04d}.png".format(scene_dir.name, frame_idx)
                if not cv2.imwrite(str(path), canvas):
                    raise OSError("无法保存驾驶数据图：{}".format(path))
                print("[vis] 已保存:", path)
        finally:
            dataset.close()
        return

    reader = SceneReader(scene_dir)
    try:
        if reader.num_frames == 0:
            print("[vis] 场景没有可视化帧：failed={} status={}，可从 LMDB 的运动学时间轴读取状态".format(
                reader.failed, reader.failure_status))
            return
        Viewer(reader, cfg.data_vis).run()
    finally:
        reader.close()


if __name__ == "__main__":
    main()
