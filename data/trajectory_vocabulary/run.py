"""几何、速度及局部二维加速度词表生成命令行入口。

模块: data/trajectory_vocabulary/run.py
依赖: argparse, config
读取配置: trajectory_vocabulary.*、model.unified_driving.nearest_weight_floor/nearest_decay_s
对外接口:
    - main() -> None
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import load_config
from data.trajectory_vocabulary import generate_action_vocabulary, generate_vocabulary


def main():
    parser = argparse.ArgumentParser(description="生成 CARLA ego 轨迹/速度词表")
    parser.add_argument("--config", default=None)
    parser.add_argument("--env", default=None)
    parser.add_argument("--actions", action="store_true", help="以最远轨迹采样生成互异动作词及标准化统计")
    args = parser.parse_args()
    cfg = load_config(args.config, args.env)
    result = (generate_action_vocabulary(cfg.trajectory_vocabulary, cfg.model.unified_driving)
              if args.actions else generate_vocabulary(cfg.trajectory_vocabulary))
    print("[vocab] scenes={} samples={} elapsed={:.1f}s".format(
        result["scene_count"], result["sample_count"], result["elapsed_s"]))


if __name__ == "__main__":
    main()
