"""Py312 采集入口 CLI：加载配置并启动采集主循环。

模块: collector/run.py
依赖: argparse, pathlib, config, collector.orchestrator
读取配置: 经 config.load_config 加载全部 carla_collector 配置
对外接口:
    - main() -> None     # 解析命令行，加载配置，调用 orchestrator.run
说明: 以脚本方式运行（python data/carla_data_collector/collector/run.py ...）。启动即把模块根与仓库根
      注入 sys.path，使 common/collector/config 等以顶层包导入，与 worker 侧保持一致。
"""

import argparse
import sys
from pathlib import Path

# 引导导入根：模块根（common/collector）与仓库根（config）
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))  # data/carla_data_collector
sys.path.insert(0, str(_HERE.parents[3]))  # 仓库根

from config import load_config
from collector.orchestrator import run


def main():
    parser = argparse.ArgumentParser(description="Carla 合成数据采集")
    parser.add_argument("--config", default=None, help="主配置文件路径（默认 config/default.yaml）")
    parser.add_argument("--env", default=None, help="环境名，叠加 config/<env>.yaml 覆盖")
    parser.add_argument("--max-scenes", type=int, default=None, help="本次最多采集场景数（覆盖配置）")
    args = parser.parse_args()

    cfg = load_config(args.config, args.env)
    run(cfg, max_scenes_override=args.max_scenes)


if __name__ == "__main__":
    main()
