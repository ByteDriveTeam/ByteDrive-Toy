"""场景和 Agent 独立生成并支持按需或预生成二值 3D 占用监督。

模块: data/driving_occupancy/driving_occupancy.py
依赖: numpy, torch, filelock, data.driving_targets, vis.data_vis.geometry
读取配置: model.driving.bev/occupancy, data.driving.fused_root
对外接口:
    - DrivingOccupancyCache(cfg) -> object
    - prepare_driving_occupancy_cache(dataset, num_workers, prefetch_factor, in_order, progress_every) -> dict
说明: 静态占用只取融合 PT 的 static.xyz；两种监督分开生成、分开存储。
      可见性按每个体素中心到相机的精确网格穿越计算，目标体素不算遮挡。
      CUDA 路径采用同构 float64 体素化和 DDA；CPU 为默认，CUDA 性能须在目标设备实测。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from filelock import FileLock
from torch.utils.data import DataLoader, Dataset

from data import driving_targets as dt
from data.driving_occupancy.checks.driving_occupancy_checks import (
    check_cache_prepare, check_fused_signature, check_fused_sources,
    check_occupancy_device)
from data.single_frame_base import resolve_repo_path
from vis.data_vis.geometry import bbox_corners, transform_matrix, transform_points, world_to_ego


__all__ = ["DrivingOccupancyCache", "prepare_driving_occupancy_cache"]


class DrivingOccupancyCache:
    """首次访问生成监督并写压缩 bitset，后续 epoch 直接读缓存。"""

    def __init__(self, cfg):
        bev = cfg.model.driving.bev
        occ = cfg.model.driving.occupancy
        self.lo = np.array((bev.x_min_m, bev.y_min_m, bev.z_min_m), dtype=np.float64)
        self.hi = np.array((bev.x_max_m, bev.y_max_m, bev.z_max_m), dtype=np.float64)
        self.step = float(occ.voxel_size_m)
        self.dims = np.rint((self.hi - self.lo) / self.step).astype(np.int64)
        self.shape = tuple(int(v) for v in self.dims[[2, 0, 1]])
        self.root = resolve_repo_path(occ.cache_dir)
        self.fused_root = resolve_repo_path(cfg.data.driving.fused_root)
        self.max_bytes = int(occ.cache_max_size_gb * 1024 ** 3)
        self.enabled = occ.cache_enabled
        self.chunk = occ.visibility_chunk
        requested = occ.compute_device
        self.device = torch.device("cuda" if requested == "auto" and torch.cuda.is_available()
                                   else "cpu" if requested == "auto" else requested)
        check_occupancy_device(self.device)
        self._fused = {}
        self._centers = None

    def check_sources(self, scene_dirs):
        """训练启动时列出缺少的 PT，不从原始数据暗中做融合。"""
        missing = [scene.name for scene in {Path(p) for p in scene_dirs}
                   if not (self.fused_root / (scene.name + ".pt")).is_file()]
        check_fused_sources(missing)

    def _payload(self, scene_dir):
        scene = Path(scene_dir).name
        if scene not in self._fused:
            pt = self.fused_root / (scene + ".pt")
            payload = torch.load(pt, map_location="cpu", weights_only=False)
            signature = payload["metadata"]["input_signature"]
            stat = (Path(scene_dir) / "lmdb" / "data.mdb").stat()
            check_fused_signature(signature, scene, stat)
            self._fused[scene] = payload
            if len(self._fused) > 1:
                self._fused.pop(next(iter(self._fused)))
        return self._fused[scene]

    def _cache_path(self, kind, scene_dir, frame_idx, fingerprint):
        digest = hashlib.sha256(json.dumps({
            "version": 2, "kind": kind, "scene": Path(scene_dir).name,
            "frame": int(frame_idx), "pt": fingerprint, "step": self.step,
            "lo": self.lo.tolist(), "hi": self.hi.tolist(),
        }, sort_keys=True).encode()).hexdigest()
        return self.root / kind / digest[:2] / (digest + ".npz")

    def contains(self, scene_dir, frame_idx):
        """检查场景与 Agent 两份缓存是否均已生成。"""
        fingerprint = self._payload(scene_dir)["metadata"]["fingerprint"]
        return all(self._cache_path(kind, scene_dir, frame_idx, fingerprint).is_file()
                   for kind in ("scene", "agent"))

    def _read_or_build(self, kind, scene_dir, frame_idx, builder):
        payload = self._payload(scene_dir)
        path = self._cache_path(kind, scene_dir, frame_idx,
                                payload["metadata"]["fingerprint"])
        if self.enabled and path.exists():
            return self._unpack(path)
        occupancy, mask = builder(payload)
        if self.enabled:
            path.parent.mkdir(parents=True, exist_ok=True)
            with FileLock(str(path) + ".lock"):
                if not path.exists():
                    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz",
                                                     delete=False) as handle:
                        temporary = Path(handle.name)
                    try:
                        np.savez_compressed(temporary, occupancy=np.packbits(
                            occupancy.ravel(), bitorder="little"),
                            mask=np.packbits(mask.ravel(), bitorder="little"))
                        with FileLock(str(self.root / ".usage.lock")):
                            state_path = self.root / ".usage.json"
                            dirty = self.root / ".usage.dirty"
                            size = (int(json.loads(state_path.read_text())["bytes"])
                                    if state_path.exists() and not dirty.exists() else sum(
                                        item.stat().st_size for item in self.root.rglob("*.npz")))
                            new_size = size + temporary.stat().st_size
                            if new_size > self.max_bytes:
                                raise RuntimeError("驾驶占用缓存达到容量上限，请增大 model.driving.occupancy.cache_max_size_gb")
                            dirty.touch()
                            os.replace(temporary, path)
                            with tempfile.NamedTemporaryFile(dir=self.root, suffix=".json",
                                                             mode="w", encoding="utf8", delete=False) as state:
                                json.dump({"bytes": new_size}, state)
                                state_temporary = Path(state.name)
                            os.replace(state_temporary, state_path)
                            dirty.unlink()
                    finally:
                        temporary.unlink(missing_ok=True)
        return occupancy, mask

    def _unpack(self, path):
        with np.load(path) as data:
            count = int(np.prod(self.shape))
            occupancy = np.unpackbits(data["occupancy"], count=count,
                                      bitorder="little").reshape(self.shape).astype(bool)
            mask = np.unpackbits(data["mask"], count=count,
                                 bitorder="little").reshape(self.shape).astype(bool)
        return occupancy, mask

    def _grid_centers(self):
        if self._centers is None:
            z, x, y = np.indices(self.shape, dtype=np.int32)
            indices = np.stack((x, y, z), -1).reshape(-1, 3)
            self._centers = self.lo + (indices + .5) * self.step
        return self._centers

    def scene(self, scene_dir, frame_idx, ego_pose, intrinsics, extrinsics, image_shape):
        """仅用 PT 静态点生成体素；遮挡判定严格不引用 Agent 或 LiDAR。"""
        def build(payload):
            points = payload["static"]["xyz"].numpy()
            # 世界轴对齐裁剪为保守球外接盒，之后再用真实刚性变换精确裁剪。
            center = np.asarray(ego_pose[:3], dtype=np.float64)
            radius = np.linalg.norm(self.hi - self.lo)
            points = points[np.all(np.abs(points - center) <= radius, axis=1)]
            occupied = (self._scene_voxels_gpu(points, ego_pose)
                        if self.device.type == "cuda" else self._scene_voxels_cpu(points, ego_pose))
            mask = self._visibility(occupied, intrinsics, extrinsics, image_shape)
            return occupied, mask
        return self._read_or_build("scene", scene_dir, frame_idx, build)

    def agent(self, scene_dir, frame_idx, ego_pose, boxes, visible,
              scene_occupied, intrinsics, extrinsics, image_shape):
        """仅对确认可见的运动框栅格化完整 OBB，含背面体素。"""
        def build(_payload):
            occupied = (torch.zeros(self.shape, dtype=torch.bool, device=self.device)
                        if self.device.type == "cuda" else np.zeros(self.shape, dtype=bool))
            ego_to_world = transform_matrix(ego_pose)
            for box, keep in zip(boxes, visible):
                if not keep:
                    continue
                world_to_box = np.linalg.inv(transform_matrix([
                    *box["location"], *box["rotation"]]))
                local_to_box = world_to_box @ ego_to_world
                corners = transform_points(bbox_corners(box), world_to_ego(ego_pose))
                low = np.maximum(np.floor((corners.min(0) - self.lo) / self.step).astype(int), 0)
                high = np.minimum(np.ceil((corners.max(0) - self.lo) / self.step).astype(int),
                                  self.dims - 1)
                if np.any(low > high):
                    continue
                x, y, z = np.mgrid[low[0]:high[0]+1, low[1]:high[1]+1, low[2]:high[2]+1]
                index = np.stack((x, y, z), -1).reshape(-1, 3)
                centers = self.lo + (index + .5) * self.step
                if self.device.type == "cuda":
                    xyz = torch.as_tensor(centers, device=self.device, dtype=torch.float64)
                    matrix = torch.as_tensor(local_to_box, device=self.device, dtype=torch.float64)
                    extent = torch.as_tensor(box["extent"], device=self.device, dtype=torch.float64)
                    inside = ((xyz @ matrix[:3, :3].T + matrix[:3, 3]).abs() <= extent).all(1)
                    selected = torch.as_tensor(index, device=self.device)[inside]
                else:
                    box_local = transform_points(centers, local_to_box)
                    inside = np.all(np.abs(box_local) <= np.asarray(box["extent"]), axis=1)
                    selected = index[inside]
                occupied[selected[:, 2], selected[:, 0], selected[:, 1]] = True
            if self.device.type == "cuda":
                occupied = occupied.cpu().numpy()
            # 负样本只来自场景与 Agent 都无遮挡的相机可见区域；正样本全框监督。
            mask = self._visibility(scene_occupied | occupied, intrinsics,
                                    extrinsics, image_shape) | occupied
            return occupied, mask
        return self._read_or_build("agent", scene_dir, frame_idx, build)

    def _scene_voxels_cpu(self, points, ego_pose):
        local = transform_points(points, world_to_ego(ego_pose))
        index = np.floor((local - self.lo) / self.step).astype(np.int64)
        valid = np.all((index >= 0) & (index < self.dims), axis=1)
        occupied = np.zeros(self.shape, dtype=bool)
        i = index[valid]
        occupied[i[:, 2], i[:, 0], i[:, 1]] = True
        return occupied

    def _scene_voxels_gpu(self, points, ego_pose):
        xyz = torch.as_tensor(points, device=self.device, dtype=torch.float64)
        matrix = torch.as_tensor(world_to_ego(ego_pose), device=self.device,
                                 dtype=torch.float64)
        low = torch.as_tensor(self.lo, device=self.device)
        dims = torch.as_tensor(self.dims, device=self.device)
        local = xyz @ matrix[:3, :3].T + matrix[:3, 3]
        index = torch.floor((local - low) / self.step).to(torch.int64)
        i = index[((index >= 0) & (index < dims)).all(1)]
        occupied = torch.zeros(self.shape, dtype=torch.bool, device=self.device)
        occupied[i[:, 2], i[:, 0], i[:, 1]] = True
        return occupied.cpu().numpy()

    def _visibility(self, occupied, intrinsics, extrinsics, image_shape):
        centers = self._grid_centers()
        height, width = image_shape
        visible = np.zeros(len(centers), dtype=bool)
        occupied_gpu = (torch.as_tensor(occupied, device=self.device)
                        if self.device.type == "cuda" else None)
        for intrinsic, extrinsic in zip(intrinsics, extrinsics):
            matrix = transform_matrix(extrinsic)
            local = transform_points(centers, np.linalg.inv(matrix))
            depth = local[:, 0]
            safe = np.maximum(depth, np.finfo(float).eps)
            u = intrinsic["fx"] * local[:, 1] / safe + intrinsic["cx"]
            v = intrinsic["cy"] - intrinsic["fy"] * local[:, 2] / safe
            in_view = (depth > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
            candidates = np.flatnonzero(in_view & ~visible)
            for indices in np.array_split(candidates, max(1, int(np.ceil(len(candidates) / self.chunk)))):
                if len(indices):
                    visible[indices] = (self._unblocked_gpu(
                        occupied_gpu, matrix[:3, 3], centers[indices])
                        if occupied_gpu is not None else self._unblocked(
                            occupied, matrix[:3, 3], centers[indices]))
        return visible.reshape(self.shape)

    def _unblocked(self, occupied, origin, targets):
        """向量化 3D DDA，只有目标之前遇到占用才判为遮挡。"""
        start = (origin - self.lo) / self.step
        target_index = np.floor((targets - self.lo) / self.step).astype(np.int64)
        delta = (targets - origin) / self.step
        step = np.sign(delta).astype(np.int64)
        safe_delta = np.where(delta == 0, 1, delta)
        current = np.broadcast_to(np.floor(start).astype(np.int64), target_index.shape).copy()
        boundary = current + (step > 0)
        tmax = np.where(step == 0, np.inf, (boundary - start) / safe_delta)
        tdelta = np.where(step == 0, np.inf, np.abs(1 / safe_delta))
        blocked = np.zeros(len(targets), dtype=bool)
        active = np.any(current != target_index, axis=1)
        while np.any(active):
            rows = np.flatnonzero(active)
            cells = current[rows]
            inside = np.all((cells >= 0) & (cells < self.dims), axis=1)
            if inside.any():
                valid_rows = rows[inside]
                xyz = current[valid_rows]
                blocked[valid_rows] |= occupied[xyz[:, 2], xyz[:, 0], xyz[:, 1]]
            active &= ~blocked
            rows = np.flatnonzero(active)
            if not len(rows):
                break
            advance = tmax[rows] == tmax[rows].min(axis=1, keepdims=True)
            current[rows] += step[rows] * advance
            tmax[rows] = np.where(advance, tmax[rows] + tdelta[rows], tmax[rows])
            active[rows] = np.any(current[rows] != target_index[rows], axis=1)
        return ~blocked

    def _unblocked_gpu(self, occupied, origin, targets):
        """与 CPU 相同的逐轴 DDA；仅将每批射线状态放在 CUDA。"""
        device = self.device
        xyz = torch.as_tensor(targets, device=device, dtype=torch.float64)
        low = torch.as_tensor(self.lo, device=device)
        origin = torch.as_tensor(origin, device=device, dtype=torch.float64)
        dims = torch.as_tensor(self.dims, device=device)
        start = (origin - low) / self.step
        target_index = torch.floor((xyz - low) / self.step).to(torch.int64)
        delta = (xyz - origin) / self.step
        step = torch.sign(delta).to(torch.int64)
        safe_delta = torch.where(delta == 0, 1, delta)
        current = torch.floor(start).to(torch.int64).expand_as(target_index).clone()
        boundary = current + (step > 0)
        inf = torch.full_like(delta, float("inf"))
        tmax = torch.where(step == 0, inf, (boundary - start) / safe_delta)
        tdelta = torch.where(step == 0, inf, (1 / safe_delta).abs())
        blocked = torch.zeros(len(targets), dtype=torch.bool, device=device)
        active = (current != target_index).any(1)
        while bool(active.any()):
            rows = torch.nonzero(active).flatten()
            cells = current[rows]
            inside = ((cells >= 0) & (cells < dims)).all(1)
            valid_rows = rows[inside]
            valid_cells = current[valid_rows]
            blocked[valid_rows] |= occupied[
                valid_cells[:, 2], valid_cells[:, 0], valid_cells[:, 1]]
            active &= ~blocked
            rows = torch.nonzero(active).flatten()
            if not len(rows):
                break
            advance = tmax[rows] == tmax[rows].min(1, keepdim=True).values
            current[rows] += step[rows] * advance
            tmax[rows] = torch.where(advance, tmax[rows] + tdelta[rows], tmax[rows])
            active[rows] = (current[rows] != target_index[rows]).any(1)
        return (~blocked).cpu().numpy()


class _OccupancyWarmupDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        self.dataset._prepare_occupancy_sample(index)
        return index


def prepare_driving_occupancy_cache(dataset, num_workers, prefetch_factor,
                                    in_order, progress_every):
    """只为缺失帧生成场景与 Agent 缓存，不构建训练模型或其他监督。"""
    check_cache_prepare(dataset, num_workers, prefetch_factor, progress_every)
    total = len(dataset)
    missing = []
    for index, (scene, frame) in enumerate(dataset.frame_index):
        if not dataset._occupancy.contains(scene, frame):
            missing.append(index)
        checked = index + 1
        if checked == 1 or checked % progress_every == 0 or checked == total:
            print("[driving-cache] scan={}/{} pending={}".format(
                checked, total, len(missing)), flush=True)
    print("[driving-cache] scan complete: existing={} pending={} train_samples={}".format(
        total - len(missing), len(missing), total), flush=True)
    # Windows worker 采用 spawn；扫描阶段最后一个场景的 PT 不应被复制到每个 worker。
    dataset._occupancy._fused.clear()
    if not missing:
        return {"existing": total, "built": 0, "total": total}
    workers = 0 if dataset._occupancy.device.type == "cuda" else num_workers
    kwargs = {"batch_size": None, "num_workers": workers,
              "persistent_workers": False}
    if workers:
        kwargs.update(prefetch_factor=prefetch_factor, in_order=in_order)
    loader = DataLoader(_OccupancyWarmupDataset(dataset, missing), **kwargs)
    for built, index in enumerate(loader, 1):
        if built == 1 or built % progress_every == 0 or built == len(missing):
            print("[driving-cache] build={}/{} frame={}/{}".format(
                built, len(missing), int(index) + 1, total), flush=True)
    return {"existing": total - len(missing), "built": len(missing), "total": total}

