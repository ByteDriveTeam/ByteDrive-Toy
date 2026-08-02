"""统一 Mesh PT 读取器：加载静态表面、动态局部模型与逐帧位姿。

模块: vis/reconstructed_mesh_vis/reader/reader.py
依赖: pathlib, torch, vis.reconstructed_mesh_vis.reader.checks
读取配置: —
对外接口:
    - ReconstructedMesh(path)
    - list_meshes(root) -> list[Path]
    - resolve_mesh(spec, root) -> Path
"""

from pathlib import Path

import torch

from vis.reconstructed_mesh_vis.reader.checks.reader_checks import (
    check_mesh_path,
    check_mesh_payload,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


class ReconstructedMesh:
    """以内存只读张量持有一个可逐帧组合的重建场景。"""

    def __init__(self, path):
        self.path = Path(path).resolve()
        check_mesh_path(self.path)
        payload = torch.load(self.path, map_location="cpu", weights_only=True)
        check_mesh_payload(payload)
        self._payload = payload
        self.static = payload["static_mesh"]
        self.dynamic = payload["dynamic_meshes"]
        self.poses = payload["dynamic_poses"]
        self.ego_pose = payload["ego_pose"]
        self.metadata = payload["metadata"]
        self.scene_name = str(self.metadata.get("scene_name", self.path.stem))
        self.frame_indices = torch.arange(len(self.ego_pose), dtype=torch.int32)
        self.method_names = tuple(self.metadata.get("method_names", ()))
        self._set_bounds()

    def _set_bounds(self):
        minima = [self.static["vertices"].min(dim=0).values]
        maxima = [self.static["vertices"].max(dim=0).values]
        if len(self.poses["transform"]):
            minima.append(self.poses["transform"][:, :3].min(dim=0).values)
            maxima.append(self.poses["transform"][:, :3].max(dim=0).values)
        lower, upper = torch.stack(minima).min(dim=0).values, torch.stack(maxima).max(dim=0).values
        self.center = ((lower + upper) * 0.5).numpy()

    @property
    def num_frames(self):
        """返回自车位姿覆盖的帧数。"""
        return len(self.ego_pose)

    @property
    def num_objects(self):
        """返回动态 actor 数，包含无几何对象。"""
        return len(self.dynamic["actor_id"])

    @property
    def num_meshed_objects(self):
        """返回具有非空 Mesh 的动态 actor 数。"""
        offsets = self.dynamic["vertex_offsets"]
        return int((offsets[1:] > offsets[:-1]).sum())


def list_meshes(root):
    """递归列出目录内的 `.mesh.pt`，并按相对路径排序。"""
    root = Path(root).resolve()
    return sorted(root.rglob("*.mesh.pt"), key=lambda path: str(path.relative_to(root))) \
        if root.is_dir() else []


def resolve_mesh(spec, root):
    """由路径、相对路径、场景名或整数索引定位 Mesh PT。"""
    root = Path(root).resolve()
    files = list_meshes(root)
    if spec is None or str(spec) == "":
        assert files, "Mesh 目录内没有 .mesh.pt: {}".format(root)
        return files[0]
    text = str(spec)
    raw = Path(text)
    candidates = ([raw] if raw.is_absolute() else
                  [_REPO_ROOT / raw, root / raw, root / (text + ".mesh.pt")])
    matches = [path.resolve() for path in candidates if path.is_file()]
    if matches:
        check_mesh_path(matches[0])
        return matches[0]
    if text.isdigit():
        index = int(text)
        assert 0 <= index < len(files), "Mesh 索引 {} 越界（共 {} 个）".format(index, len(files))
        return files[index]
    stem_matches = [path for path in files if path.name == text or path.stem == text
                    or path.name == text + ".mesh.pt"]
    if len(stem_matches) == 1:
        return stem_matches[0]
    raise FileNotFoundError("无法唯一定位重建 Mesh: {}（搜索目录 {}）".format(spec, root))


__all__ = ["ReconstructedMesh", "list_meshes", "resolve_mesh"]
