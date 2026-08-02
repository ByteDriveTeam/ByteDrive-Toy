"""融合重建点云可视化 CLI：选择 PT 场景并启动 Open3D 交互窗口。

模块: vis/reconstructed_pointcloud_vis/run.py
依赖: argparse, pathlib, sys, config, vis.reconstructed_pointcloud_vis.reader/viewer
读取配置: reconstructed_pointcloud_vis.pointcloud_dir/file；其余配置透传给查看器
对外接口:
    - main() -> None
说明: 从项目根运行 ``python -m vis.reconstructed_pointcloud_vis.run --input <文件/场景名/索引>``。
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import load_config
from vis.reconstructed_pointcloud_vis.reader import (
    FusionPointcloud,
    list_pointclouds,
    resolve_pointcloud,
)
from vis.reconstructed_pointcloud_vis.viewer import PointcloudViewer


def _resolve(path):
    value = Path(path)
    return value.resolve() if value.is_absolute() else (_REPO_ROOT / value).resolve()


def main():
    """解析参数、加载重建点云并进入 Open3D 交互窗口。"""
    parser = argparse.ArgumentParser(description="ByteDrive 融合重建点云 Open3D 可视化")
    parser.add_argument("--config", default=None, help="主配置文件路径（缺省 config/default.yaml）")
    parser.add_argument("--env", default=None, help="环境覆盖名（叠加 config/<env>.yaml）")
    parser.add_argument("--input", default=None, help="PT 路径、场景名或目录内整数索引")
    parser.add_argument("--list", action="store_true", help="列出点云目录内的 PT 后退出")
    args = parser.parse_args()

    cfg = load_config(args.config, args.env)
    vcfg = cfg.reconstructed_pointcloud_vis
    root = _resolve(vcfg.pointcloud_dir)
    if args.list:
        files = list_pointclouds(root)
        print("\n".join("{:4d}  {}".format(index, path.name)
                        for index, path in enumerate(files)) or "目录内没有 .pt: {}".format(root))
        return
    spec = args.input if args.input is not None else vcfg.file
    path = resolve_pointcloud(spec, root)
    print("[重建点云] 加载:", path)
    data = FusionPointcloud(path)
    PointcloudViewer(data, vcfg, cfg.model.driving.bev).run()


if __name__ == "__main__":
    main()
