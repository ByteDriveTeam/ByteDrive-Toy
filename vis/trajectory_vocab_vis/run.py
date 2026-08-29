"""轨迹词表 PNG 可视化命令行入口。

模块: vis/trajectory_vocab_vis/run.py
依赖: argparse, config, vis.trajectory_vocab_vis
读取配置: trajectory_vocabulary.*
对外接口:
    - main() -> None
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import load_config
from vis.trajectory_vocab_vis.trajectory_vocab_vis import render_vocabulary


def main():
    parser = argparse.ArgumentParser(description="可视化轨迹与速度词表")
    parser.add_argument("--config", default=None); parser.add_argument("--env", default=None)
    args = parser.parse_args(); cfg = load_config(args.config, args.env).trajectory_vocabulary
    for path in render_vocabulary(cfg): print("[vis]", path)


if __name__ == "__main__":
    main()
