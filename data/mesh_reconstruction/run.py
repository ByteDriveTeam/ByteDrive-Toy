"""融合点云稀疏 TUDF/可选 Mesh 批处理 CLI。

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
from data.mesh_reconstruction.udf import run_udf_reconstruction


def main():
    """解析 CLI、批量重建并按汇总状态设置退出码。"""
    parser = argparse.ArgumentParser(description="ByteDrive 融合点云稀疏 TUDF/可选 Mesh 重建")
    parser.add_argument("--config", default=None, help="主配置文件路径")
    parser.add_argument("--env", default=None, help="环境覆盖名")
    parser.add_argument("--input", default=None, help="融合 PT 或递归目录")
    parser.add_argument("--output", default=None, help="项目内输出目录")
    parser.add_argument("--force", action="store_true", help="忽略已有同指纹结果并强制重建")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--mesh", action="store_true", help="显式导出 Poisson Mesh")
    mode.add_argument("--udf", action="store_true", help="显式生成稀疏 TUDF")
    args = parser.parse_args()
    cfg = load_config(args.config, args.env)
    output_format = "mesh" if args.mesh else "sparse_udf" if args.udf \
        else cfg.mesh_reconstruction.output_format
    runner = run_reconstruction if output_format == "mesh" else run_udf_reconstruction
    report = runner(cfg, args.input, args.output, args.force)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
