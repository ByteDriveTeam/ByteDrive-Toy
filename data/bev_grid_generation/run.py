"""离线 10 图层 BEV 栅格生成 CLI。

模块: data/bev_grid_generation/run.py
依赖: argparse, config.load_config, data.bev_grid_generation
读取配置: data.bev_grid_generation（实现读取全部细项）
对外接口:
    - main(argv=None) -> None
"""

import argparse

from config import load_config
from data.bev_grid_generation import generate_bev_grids


def main(argv=None) -> None:
    """加载配置并按场景并行生成离线栅格。"""
    parser = argparse.ArgumentParser(description="生成 ByteDrive 10 图层特权 BEV 栅格")
    parser.add_argument("--config", default=None, help="主配置文件路径")
    parser.add_argument("--env", default=None, help="config/<env>.yaml 覆盖名")
    parser.add_argument("--scene-limit", type=int, default=None, help="仅处理前 N 个场景")
    args = parser.parse_args(argv)
    stats = generate_bev_grids(load_config(args.config, args.env), args.scene_limit)
    print("[bev-grid] scenes={scenes} generated={generated} skipped={skipped} frames={frames}".format(**stats))


if __name__ == "__main__":
    main()
