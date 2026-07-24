"""CARLA 0.9.15 行为克隆闭环 CLI：加载配置并启动异构 episode 编排。

模块: clone_loop/run.py
依赖: argparse, pathlib, sys, config.load_config, clone_loop.run_closed_loop
读取配置: 经 config.load_config 加载 clone_loop、model 与 data.dataset 相关配置
对外接口:
    - main(argv=None) -> None
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from clone_loop import run_closed_loop
from config import load_config


def main(argv=None):
    """解析闭环 CLI 参数并运行路线队列。"""
    parser = argparse.ArgumentParser(description="ByteDrive CARLA 0.9.15 闭环驾驶")
    parser.add_argument("--config", default=None, help="主配置文件路径")
    parser.add_argument("--env", default=None, help="叠加 config/<env>.yaml")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="覆盖 clone_loop.route.max_episodes；0 表示全部")
    args = parser.parse_args(argv)
    run_closed_loop(
        load_config(args.config, args.env),
        max_episodes_override=args.max_episodes)


if __name__ == "__main__":
    main()
