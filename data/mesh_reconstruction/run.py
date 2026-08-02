"""融合点云 Mesh 批处理 CLI，支持可选水密修复。

模块: data/mesh_reconstruction/run.py
依赖: argparse, json, config, data.mesh_reconstruction
读取配置: mesh_reconstruction 全树；--input/--output 仅覆盖运行时路径
对外接口:
    - main() -> None
说明: 从项目根运行 `python -m data.mesh_reconstruction.run`；失败详情写入 mesh_report.json。
"""

import argparse
import json

from config import load_config
from data.mesh_reconstruction import run_reconstruction


def main():
    """解析 CLI、批量重建并按汇总状态设置退出码。"""
    parser = argparse.ArgumentParser(description="ByteDrive 融合点云 Mesh 重建")
    parser.add_argument("--config", default=None, help="主配置文件路径")
    parser.add_argument("--env", default=None, help="环境覆盖名")
    parser.add_argument("--input", default=None, help="融合 PT 或递归目录")
    parser.add_argument("--output", default=None, help="项目内输出目录")
    parser.add_argument("--force", action="store_true", help="忽略已有同指纹结果并强制重建")
    args = parser.parse_args()
    report = run_reconstruction(
        load_config(args.config, args.env), args.input, args.output, args.force)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
