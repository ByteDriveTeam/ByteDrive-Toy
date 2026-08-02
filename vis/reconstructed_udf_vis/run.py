"""稀疏 TUDF Open3D 逐帧可视化 CLI。

模块: vis/reconstructed_udf_vis/run.py
依赖: argparse, pathlib, config, vis.reconstructed_udf_vis.reader/viewer
读取配置: reconstructed_udf_vis 全树；model.driving.bev
对外接口:
    - main() -> None
"""

import argparse
from pathlib import Path

from config import load_config
from vis.reconstructed_udf_vis.reader import ReconstructedUdf, list_udfs, resolve_udf
from vis.reconstructed_udf_vis.viewer import UdfViewer

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve(path):
    value = Path(path)
    return value.resolve() if value.is_absolute() else (_REPO_ROOT / value).resolve()


def main():
    parser = argparse.ArgumentParser(description="ByteDrive 稀疏 TUDF 逐帧可视化")
    parser.add_argument("--config", default=None, help="主配置文件路径")
    parser.add_argument("--env", default=None, help="环境覆盖名")
    parser.add_argument("--input", default=None, help="TUDF PT、场景名或整数索引")
    parser.add_argument("--list", action="store_true", help="列出 TUDF 后退出")
    args = parser.parse_args()
    cfg = load_config(args.config, args.env)
    visual = cfg.reconstructed_udf_vis
    root = _resolve(visual.udf_dir)
    if args.list:
        files = list_udfs(root)
        print("\n".join("{:4d}  {}".format(index, path.relative_to(root))
                        for index, path in enumerate(files))
              or "目录内没有 .udf.pt: {}".format(root))
        return
    path = resolve_udf(args.input if args.input is not None else visual.file, root)
    print("[稀疏TUDF] 加载:", path)
    UdfViewer(ReconstructedUdf(path), visual, cfg.model.driving.bev).run()


if __name__ == "__main__":
    main()
