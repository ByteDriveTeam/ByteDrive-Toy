"""离线 10 图层 BEV 栅格保存与播放 CLI。

模块: vis/bev_grid_vis/run.py
依赖: argparse, config.load_config, vis.bev_grid_vis
读取配置: bev_grid_vis（实现读取全部细项）
对外接口:
    - main(argv=None) -> None
"""

import argparse

from config import load_config
from vis.bev_grid_vis import visualize_bev_grid


def main(argv=None) -> None:
    """保存指定帧画布，并可选启动时序窗口。"""
    parser = argparse.ArgumentParser(description="可视化 ByteDrive 10 图层 BEV 栅格")
    parser.add_argument("--config", default=None, help="主配置文件路径")
    parser.add_argument("--env", default=None, help="config/<env>.yaml 覆盖名")
    parser.add_argument("--scene", default=None, help="场景目录名")
    parser.add_argument("--frame", type=int, default=None, help="起始帧")
    parser.add_argument("--output", default=None, help="单帧 PNG 输出路径")
    parser.add_argument("--show", action="store_true", help="显示顺序播放窗口")
    args = parser.parse_args(argv)
    output = visualize_bev_grid(
        load_config(args.config, args.env), args.scene, args.frame, args.output, args.show)
    if output is not None:
        print("[bev-grid-vis] {}".format(output))


if __name__ == "__main__":
    main()
