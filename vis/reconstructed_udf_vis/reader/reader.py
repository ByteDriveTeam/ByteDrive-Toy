"""统一稀疏 TUDF PT 读取器。

模块: vis/reconstructed_udf_vis/reader/reader.py
依赖: pathlib, torch, data.mesh_reconstruction.udf.checks
读取配置: —
对外接口:
    - ReconstructedUdf(path)
    - list_udfs(root) -> list[Path]
    - resolve_udf(spec, root) -> Path
"""

from pathlib import Path

import torch

from data.mesh_reconstruction.udf.checks.udf_checks import check_udf_payload
from vis.reconstructed_udf_vis.reader.checks.reader_checks import check_udf_path

_REPO_ROOT = Path(__file__).resolve().parents[3]


class ReconstructedUdf:
    """以内存 CPU 张量持有静态世界场、动态局部场与逐帧位姿。"""

    def __init__(self, path):
        self.path = Path(path).resolve()
        check_udf_path(self.path)
        payload = torch.load(self.path, map_location="cpu", weights_only=True)
        check_udf_payload(payload, verify_unique=False)
        self.static = payload["static_udf"]
        self.dynamic = payload["dynamic_udfs"]
        self.poses = payload["dynamic_poses"]
        self.ego_pose = payload["ego_pose"]
        self.metadata = payload["metadata"]
        self.scene_name = str(self.metadata.get("scene_name", self.path.stem))
        self.frame_indices = torch.arange(len(self.ego_pose), dtype=torch.int32)
        coords = self.static["voxel_coords"].to(torch.float32)
        size = self.static["voxel_size_m"]
        centers = (coords + 0.5) * size
        self.center = ((centers.amin(0) + centers.amax(0)) * 0.5).numpy()

    @property
    def num_frames(self):
        return len(self.ego_pose)

    @property
    def num_objects(self):
        return len(self.dynamic["actor_id"])


def list_udfs(root):
    root = Path(root).resolve()
    return sorted(root.rglob("*.udf.pt"), key=lambda path: str(path.relative_to(root))) \
        if root.is_dir() else []


def resolve_udf(spec, root):
    root = Path(root).resolve()
    files = list_udfs(root)
    if spec is None or str(spec) == "":
        assert files, "TUDF 目录内没有 .udf.pt: {}".format(root)
        return files[0]
    text, raw = str(spec), Path(str(spec))
    candidates = [raw] if raw.is_absolute() else [
        _REPO_ROOT / raw, root / raw, root / (text + ".udf.pt")]
    matches = [path.resolve() for path in candidates if path.is_file()]
    if matches:
        check_udf_path(matches[0])
        return matches[0]
    if text.isdigit():
        index = int(text)
        assert 0 <= index < len(files), "TUDF 索引越界"
        return files[index]
    matches = [path for path in files if path.name == text or path.stem == text]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError("无法唯一定位 TUDF: {}".format(spec))


__all__ = ["ReconstructedUdf", "list_udfs", "resolve_udf"]
