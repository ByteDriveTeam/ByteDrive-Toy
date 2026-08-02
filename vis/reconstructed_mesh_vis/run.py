"""水密 Mesh 可视化 CLI：选择统一 PT 并启动 Open3D 逐帧查看器。

模块: vis/reconstructed_mesh_vis/run.py
依赖: argparse, pathlib, sys, config, vis.reconstructed_mesh_vis.reader/viewer
读取配置: reconstructed_mesh_vis 全树
对外接口:
    - main() -> None
"""

import argparse
from pathlib import Path

from config import load_config
from vis.reconstructed_mesh_vis.reader import ReconstructedMesh, list_meshes, resolve_mesh
from vis.reconstructed_mesh_vis.viewer import MeshViewer

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve(path):
    value = Path(path)
    return value.resolve() if value.is_absolute() else (_REPO_ROOT / value).resolve()


def main():
    """解析参数、加载 Mesh 并进入 Open3D 交互窗口。"""
    parser = argparse.ArgumentParser(description="ByteDrive 水密 Mesh 逐帧可视化")
    parser.add_argument("--config", default=None, help="主配置文件路径")
    parser.add_argument("--env", default=None, help="环境覆盖名")
    parser.add_argument("--input", default=None, help="Mesh PT 路径、场景名或整数索引")
    parser.add_argument("--list", action="store_true", help="列出 Mesh 后退出")
    args = parser.parse_args()
    cfg = load_config(args.config, args.env)
    visual = cfg.reconstructed_mesh_vis
    root = _resolve(visual.mesh_dir)
    if args.list:
        files = list_meshes(root)
        print("\n".join("{:4d}  {}".format(index, path.relative_to(root))
                        for index, path in enumerate(files))
              or "目录内没有 .mesh.pt: {}".format(root))
        return
    path = resolve_mesh(args.input if args.input is not None else visual.file, root)
    print("[重建Mesh] 加载:", path)
    MeshViewer(ReconstructedMesh(path), visual, cfg.model.driving.bev).run()


if __name__ == "__main__":
    main()
