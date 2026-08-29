"""轨迹与速度词表生成命令行入口。

模块: data/trajectory_vocabulary/run.py
依赖: argparse, config
读取配置: trajectory_vocabulary.*
对外接口:
    - main() -> None
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import load_config
from data.trajectory_vocabulary import generate_vocabulary


def main():
    parser = argparse.ArgumentParser(description="生成 CARLA ego 轨迹/速度词表")
    parser.add_argument("--config", default=None)
    parser.add_argument("--env", default=None)
    args = parser.parse_args()
    result = generate_vocabulary(load_config(args.config, args.env).trajectory_vocabulary)
    print("[vocab] scenes={} samples={} elapsed={:.1f}s".format(
        result["scene_count"], result["sample_count"], result["elapsed_s"]))


if __name__ == "__main__":
    main()
