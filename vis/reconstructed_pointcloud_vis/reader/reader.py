"""融合重建点云读取器：加载统一 PT 格式并提供自车位姿、来源与 actor 索引。

模块: vis/reconstructed_pointcloud_vis/reader/reader.py
依赖: pathlib, numpy, torch, vis.reconstructed_pointcloud_vis.reader.checks
读取配置: —
对外接口:
    - FusionPointcloud(path)                         # 内存只读点云与统计索引
    - list_pointclouds(root) -> list[Path]           # 列出目录直属融合产物
    - resolve_pointcloud(spec, root) -> Path         # 由路径/场景名/索引定位 PT
"""

from pathlib import Path

import numpy as np
import torch

from vis.reconstructed_pointcloud_vis.reader.checks import (
    check_pointcloud_path,
    check_pointcloud_payload,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


class FusionPointcloud:
    """加载一个场景的融合重建点云。

    参数:
        path: 融合系统输出的 ``<场景名>.pt`` 文件
    说明:
        点级张量转为共享 CPU 内存的 numpy 视图；不会复制百万级点坐标。
    """

    def __init__(self, path):
        self.path = Path(path).resolve()
        check_pointcloud_path(self.path)
        payload = torch.load(self.path, map_location="cpu", weights_only=True)
        check_pointcloud_payload(payload)
        tensors = {name: payload[name].contiguous() for name in
                   ("xyz", "obj_tag", "source", "actor_id", "frame_index", "ego_pose")}
        self._tensors = tensors
        self.xyz = tensors["xyz"].numpy()
        self.obj_tag = tensors["obj_tag"].numpy()
        self.source = tensors["source"].numpy()
        self.actor_id = tensors["actor_id"].numpy()
        self.frame_index = tensors["frame_index"].numpy()
        self.ego_pose = tensors["ego_pose"].numpy()
        self.metadata = dict(payload["metadata"])
        self.scene_name = str(self.metadata.get("scene_name", self.path.stem))
        dynamic = self.source == 1
        self.dynamic_frames = np.unique(self.frame_index[dynamic])
        self.frame_indices = np.arange(len(self.ego_pose), dtype=np.int32)
        self.actor_ids = np.unique(self.actor_id[dynamic])
        self._num_static = int(np.count_nonzero(~dynamic))
        self._num_dynamic = int(np.count_nonzero(dynamic))
        if len(self.xyz):
            self.center = (self.xyz.min(axis=0) + self.xyz.max(axis=0)) * 0.5
            self.height_range = np.asarray(
                (self.xyz[:, 2].min(), self.xyz[:, 2].max()), dtype=np.float32)
        else:
            self.center = np.zeros(3, dtype=np.float32)
            self.height_range = np.array((0.0, 1.0), dtype=np.float32)

    @property
    def num_points(self):
        """返回总点数。"""
        return len(self.xyz)

    @property
    def num_static(self):
        """返回静态点数。"""
        return self._num_static

    @property
    def num_dynamic(self):
        """返回动态点数（含各帧重复放置的完整对象模型）。"""
        return self._num_dynamic

    @property
    def num_frames(self):
        """返回自车位姿覆盖的传感器帧数。"""
        return len(self.ego_pose)


def list_pointclouds(root):
    """按文件名列出目录直属的融合 PT 文件。"""
    root = Path(root).resolve()
    return sorted(root.glob("*.pt"), key=lambda path: path.name) if root.is_dir() else []


def resolve_pointcloud(spec, root):
    """由文件路径、场景名或整数索引定位一个融合 PT 文件。

    参数:
        spec: 文件路径、场景名、文件名或目录内索引；空值取首个文件
        root: 融合产物搜索目录
    返回:
        已解析的绝对 PT 路径
    """
    root = Path(root).resolve()
    files = list_pointclouds(root)
    if spec is None or str(spec) == "":
        assert files, "点云目录内没有 .pt 文件: {}".format(root)
        return files[0]
    text = str(spec)
    raw = Path(text)
    candidates = ([raw] if raw.is_absolute() else
                  [_REPO_ROOT / raw, root / raw, root / (text + ".pt")])
    matches = [path.resolve() for path in candidates if path.is_file()]
    if matches:
        check_pointcloud_path(matches[0])
        return matches[0]
    if text.isdigit():
        index = int(text)
        assert 0 <= index < len(files), "点云索引 {} 越界（共 {} 个）".format(index, len(files))
        return files[index]
    raise FileNotFoundError("无法定位重建点云: {}（搜索目录 {}）".format(spec, root))


__all__ = ["FusionPointcloud", "list_pointclouds", "resolve_pointcloud"]
