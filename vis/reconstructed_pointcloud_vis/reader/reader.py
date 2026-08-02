"""融合重建点云读取器：加载分离的静态地图、动态对象模型及逐帧位姿。

模块: vis/reconstructed_pointcloud_vis/reader/reader.py
依赖: pathlib, numpy, torch, vis.reconstructed_pointcloud_vis.reader.checks
读取配置: —
对外接口:
    - FusionPointcloud(path)                         # 内存只读的规范化场景点云
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
    """加载一个场景的静态地图、对象局部模型与逐帧世界位姿。"""

    def __init__(self, path):
        self.path = Path(path).resolve()
        check_pointcloud_path(self.path)
        payload = torch.load(self.path, map_location="cpu", weights_only=True)
        check_pointcloud_payload(payload)
        self._payload = payload
        static = payload["static"]
        objects = payload["dynamic_objects"]
        poses = payload["dynamic_poses"]
        self.static_xyz = static["xyz"].contiguous().numpy()
        self.static_obj_tag = static["obj_tag"].contiguous().numpy()
        self.object_actor_id = objects["actor_id"].contiguous().numpy()
        self.object_class_id = objects["class_id"].contiguous().numpy()
        self.object_extent = objects["extent"].contiguous().numpy()
        self.object_point_offsets = objects["point_offsets"].contiguous().numpy()
        self.dynamic_xyz_local = objects["xyz_local"].contiguous().numpy()
        self.dynamic_obj_tag = objects["obj_tag"].contiguous().numpy()
        self.pose_object_index = poses["object_index"].contiguous().numpy()
        self.pose_frame_index = poses["frame_index"].contiguous().numpy()
        self.pose_transform = poses["transform"].contiguous().numpy()
        self.ego_pose = payload["ego_pose"].contiguous().numpy()
        self.metadata = dict(payload["metadata"])
        self.scene_name = str(self.metadata.get("scene_name", self.path.stem))
        self.frame_indices = np.arange(len(self.ego_pose), dtype=np.int32)
        self.dynamic_frames = np.unique(self.pose_frame_index)
        self.actor_ids = self.object_actor_id
        self._set_bounds()

    def _set_bounds(self):
        minima = [self.static_xyz.min(axis=0)] if len(self.static_xyz) else []
        maxima = [self.static_xyz.max(axis=0)] if len(self.static_xyz) else []
        if len(self.pose_transform):
            minima.append(self.pose_transform[:, :3].min(axis=0))
            maxima.append(self.pose_transform[:, :3].max(axis=0))
        if minima:
            lower = np.stack(minima).min(axis=0)
            upper = np.stack(maxima).max(axis=0)
            self.center = (lower + upper) * 0.5
            self.height_range = np.asarray((lower[2], upper[2]), dtype=np.float32)
        else:
            self.center = np.zeros(3, dtype=np.float32)
            self.height_range = np.asarray((0.0, 1.0), dtype=np.float32)

    @property
    def num_points(self):
        """返回静态点与动态规范模型点的总数，不含逐帧重复放置。"""
        return self.num_static + self.num_dynamic

    @property
    def num_static(self):
        """返回只存一次的静态地图点数。"""
        return len(self.static_xyz)

    @property
    def num_dynamic(self):
        """返回只存一次的动态对象规范模型点数。"""
        return len(self.dynamic_xyz_local)

    @property
    def num_objects(self):
        """返回动态对象记录数，包括暂未重建出点的 Box actor。"""
        return len(self.object_actor_id)

    @property
    def num_poses(self):
        """返回动态对象逐帧位姿记录数。"""
        return len(self.pose_frame_index)

    @property
    def num_frames(self):
        """返回自车位姿覆盖的传感器帧数。"""
        return len(self.ego_pose)


def list_pointclouds(root):
    """按文件名列出目录直属的融合 PT 文件。"""
    root = Path(root).resolve()
    return sorted(root.glob("*.pt"), key=lambda path: path.name) if root.is_dir() else []


def resolve_pointcloud(spec, root):
    """由文件路径、场景名或整数索引定位一个融合 PT 文件。"""
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
