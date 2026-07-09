"""可视化入口 CLI：加载配置、定位场景目录、启动交互窗口。

模块: vis/data_vis/run.py
依赖: argparse, sys, pathlib, config.load_config, vis.data_vis.reader, vis.data_vis.viewer
读取配置: data_vis.scene_root（默认场景搜索根）；样式参数经 cfg.data_vis 透传给 reader/viewer/draw
对外接口:
    - main() -> None     # 解析命令行并启动可视化
说明: --scene 可为目录路径、场景名(scene_000000) 或在 scene_root 下的整数索引；缺省取 scene_root 下第一个。
      从仓库根运行：./.venv/Scripts/python.exe vis/data_vis/run.py [--scene N] [--config ...] [--env ...]
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import load_config
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
    args = parser.parse_args()

    cfg = load_config(args.config, args.env)
    scene_root = _resolve(cfg.data_vis.scene_root)
    scene_dir = _pick_scene(args.scene, scene_root)
    print("[vis] 打开场景:", scene_dir)

    reader = SceneReader(scene_dir)
    try:
        Viewer(reader, cfg.data_vis).run()
    finally:
        reader.close()


if __name__ == "__main__":
    main()
