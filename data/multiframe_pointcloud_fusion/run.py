"""多帧点云融合 CLI：加载配置并处理单场景或递归数据集。

模块: data/multiframe_pointcloud_fusion/run.py
依赖: argparse, json, config.load_config, data.multiframe_pointcloud_fusion
读取配置: multiframe_pointcloud_fusion 全树；--input/--output 仅覆盖运行时路径
对外接口:
    - main() -> None
说明: 从仓库根运行 `python -m data.multiframe_pointcloud_fusion.run`；存在失败场景时处理完
      其余场景后以非零状态退出，详细原因写入输出目录 fusion_report.json。
"""

import argparse
import json

from config import load_config
from data.multiframe_pointcloud_fusion import run_fusion


def main():
    """解析 CLI、执行融合并按汇总结果设置退出状态。"""
    parser = argparse.ArgumentParser(description="ByteDrive 多帧语义点云融合")
    parser.add_argument("--config", default=None, help="主配置文件路径（缺省 config/default.yaml）")
    parser.add_argument("--env", default=None, help="环境覆盖名（叠加 config/<env>.yaml）")
    parser.add_argument("--input", default=None, help="单场景或数据集根目录（缺省读取配置）")
    parser.add_argument("--output", default=None, help="项目内输出目录（缺省读取配置）")
    args = parser.parse_args()
    report = run_fusion(
        load_config(args.config, args.env), input_path=args.input, output_dir=args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
