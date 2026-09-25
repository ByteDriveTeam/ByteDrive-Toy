"""场景和 Agent 独立生成并支持按需或预生成二值 3D 占用监督。

模块: data/driving_occupancy/driving_occupancy.py
依赖: numpy, torch, filelock, 可选 numba, vis.data_vis.geometry
读取配置: model.driving.bev/occupancy, data.driving.fused_root
对外接口:
    - DrivingOccupancyCache(cfg) -> object
    - prepare_driving_occupancy_cache(dataset, num_workers, prefetch_factor, in_order, progress_every) -> dict
说明: 静态占用只取融合 PT 的 static.xyz；两种监督分开生成、分开存储。
      Agent 缓存同时保存逐框可见性，供 Detect 监督复用；旧 Agent 缓存按版本自动重建。
      可见性按每个体素中心到相机的精确网格穿越计算，目标体素不算遮挡。
      固定标定的完整 DDA 路径预计算后由 CPU/GPU 共用；超限回退逐射线 DDA。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import warnings

import numpy as np
import torch
from filelock import FileLock
from torch.utils.data import DataLoader, Dataset

try:
    from numba import config as numba_config, njit, prange, set_num_threads
except ImportError:
    njit = None
    prange = range

from data.driving_occupancy.checks.driving_occupancy_checks import (
    check_cache_prepare, check_fused_signature, check_fused_sources,
    check_occupancy_device)
from data.single_frame_base import resolve_repo_path
from vis.data_vis.geometry import bbox_corners, transform_matrix, transform_points, world_to_ego


__all__ = ["DrivingOccupancyCache", "prepare_driving_occupancy_cache"]


if njit is not None:
    @njit(cache=True)
    def _ray_path(origin, target, low, step, dims, output, offset):
        """列出目标体素之前的网格单元；越界单元可省略，因为永远不占用。"""
        sx, sy, sz = (origin[0] - low[0]) / step, (origin[1] - low[1]) / step, (origin[2] - low[2]) / step
        tx = int(np.floor((target[0] - low[0]) / step))
        ty = int(np.floor((target[1] - low[1]) / step))
        tz = int(np.floor((target[2] - low[2]) / step))
        vx, vy, vz = (target[0] - origin[0]) / step, (target[1] - origin[1]) / step, (target[2] - origin[2]) / step
        dx = 1 if vx > 0 else -1 if vx < 0 else 0
        dy = 1 if vy > 0 else -1 if vy < 0 else 0
        dz = 1 if vz > 0 else -1 if vz < 0 else 0
        cx, cy, cz = int(np.floor(sx)), int(np.floor(sy)), int(np.floor(sz))
        mx = np.inf if dx == 0 else (cx + (dx > 0) - sx) / vx
        my = np.inf if dy == 0 else (cy + (dy > 0) - sy) / vy
        mz = np.inf if dz == 0 else (cz + (dz > 0) - sz) / vz
        ix = np.inf if dx == 0 else abs(1.0 / vx)
        iy = np.inf if dy == 0 else abs(1.0 / vy)
        iz = np.inf if dz == 0 else abs(1.0 / vz)
        count = 0
        while cx != tx or cy != ty or cz != tz:
            if 0 <= cx < dims[0] and 0 <= cy < dims[1] and 0 <= cz < dims[2]:
                if len(output):
                    output[offset + count] = cz * dims[0] * dims[1] + cx * dims[1] + cy
                count += 1
            minimum = min(mx, my, mz)
            if mx == minimum:
                cx += dx
                mx += ix
            if my == minimum:
                cy += dy
                my += iy
            if mz == minimum:
                cz += dz
                mz += iz
        return count


    @njit(parallel=True, cache=True)
    def _ray_path_counts(origin, targets, low, step, dims):
        counts = np.empty(len(targets), dtype=np.int32)
        empty = np.empty(0, dtype=np.uint32)
        for row in prange(len(targets)):
            counts[row] = _ray_path(origin, targets[row], low, step, dims, empty, 0)
        return counts


    @njit(parallel=True, cache=True)
    def _ray_path_fill(origin, targets, low, step, dims, offsets, indices):
        for row in prange(len(targets)):
            _ray_path(origin, targets[row], low, step, dims, indices, offsets[row])


    @njit(parallel=True, cache=True)
    def _lookup_visible_cpu(flat_occupied, offsets, indices):
        visible = np.ones(len(offsets) - 1, dtype=np.bool_)
        for row in prange(len(visible)):
            for index in range(offsets[row], offsets[row + 1]):
                if flat_occupied[indices[index]]:
                    visible[row] = False
                    break
        return visible


    @njit(parallel=True, cache=True)
    def _trace_rays_cpu(occupied, origin, targets, low, step, dims):
        """每条射线独立执行与 NumPy 路径相同的精确 3D DDA。"""
        result = np.ones(len(targets), dtype=np.bool_)
        sx, sy, sz = (origin[0] - low[0]) / step, (origin[1] - low[1]) / step, (origin[2] - low[2]) / step
        for row in prange(len(targets)):
            tx = int(np.floor((targets[row, 0] - low[0]) / step))
            ty = int(np.floor((targets[row, 1] - low[1]) / step))
            tz = int(np.floor((targets[row, 2] - low[2]) / step))
            vx = (targets[row, 0] - origin[0]) / step
            vy = (targets[row, 1] - origin[1]) / step
            vz = (targets[row, 2] - origin[2]) / step
            dx = 1 if vx > 0 else -1 if vx < 0 else 0
            dy = 1 if vy > 0 else -1 if vy < 0 else 0
            dz = 1 if vz > 0 else -1 if vz < 0 else 0
            cx, cy, cz = int(np.floor(sx)), int(np.floor(sy)), int(np.floor(sz))
            mx = np.inf if dx == 0 else (cx + (dx > 0) - sx) / vx
            my = np.inf if dy == 0 else (cy + (dy > 0) - sy) / vy
            mz = np.inf if dz == 0 else (cz + (dz > 0) - sz) / vz
            ix = np.inf if dx == 0 else abs(1.0 / vx)
            iy = np.inf if dy == 0 else abs(1.0 / vy)
            iz = np.inf if dz == 0 else abs(1.0 / vz)
            while cx != tx or cy != ty or cz != tz:
                if 0 <= cx < dims[0] and 0 <= cy < dims[1] and 0 <= cz < dims[2]:
                    if occupied[cz, cx, cy]:
                        result[row] = False
                        break
                minimum = min(mx, my, mz)
                if mx == minimum:
                    cx += dx
                    mx += ix
                if my == minimum:
                    cy += dy
                    my += iy
                if mz == minimum:
                    cz += dz
                    mz += iz
        return result


class DrivingOccupancyCache:
    """首次访问生成监督与 Agent 可见性，后续 epoch 直接读压缩缓存。"""

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
        self.cpu_threads = occ.cpu_threads
        self.ray_lookup_enabled = occ.ray_lookup_enabled
        self.ray_lookup_max_bytes = int(occ.ray_lookup_max_size_gb * 1024 ** 3)
        if self.ray_lookup_enabled and njit is None:
            warnings.warn("驾驶占用路径查表需要 Numba；当前退回逐射线 DDA，请安装 numba>=0.67,<0.68。",
                          RuntimeWarning, stacklevel=2)
        self._jit_pid = None
        requested = occ.compute_device
        self.device = torch.device("cuda" if requested == "auto" and torch.cuda.is_available()
                                   else "cpu" if requested == "auto" else requested)
        check_occupancy_device(self.device)
        self._fused = {}
        self._centers = None
        self._view_key = None
        self._view_candidates = None
        self._ray_tables = None
        self._ray_gpu_tables = None
        self._gpu_static_scene = None
        self._gpu_static_points = None

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
            "version": 3 if kind == "agent" else 2, "kind": kind,
            "scene": Path(scene_dir).name,
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
            return self._unpack(path, kind)
        result = builder(payload)
        occupancy, mask = result[:2]
        extra = ({"visible": np.packbits(result[2], bitorder="little"),
                  "visible_count": np.asarray(len(result[2]), dtype=np.int32)}
                 if kind == "agent" else {})
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
                            mask=np.packbits(mask.ravel(), bitorder="little"), **extra)
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
        return result

    def _unpack(self, path, kind):
        with np.load(path) as data:
            count = int(np.prod(self.shape))
            occupancy = np.unpackbits(data["occupancy"], count=count,
                                      bitorder="little").reshape(self.shape).astype(bool)
            mask = np.unpackbits(data["mask"], count=count,
                                 bitorder="little").reshape(self.shape).astype(bool)
            if kind == "agent":
                visible = np.unpackbits(data["visible"],
                                        count=int(data["visible_count"]),
                                        bitorder="little").astype(bool)
        return (occupancy, mask, visible) if kind == "agent" else (occupancy, mask)

    def _grid_centers(self):
        if self._centers is None:
            z, x, y = np.indices(self.shape, dtype=np.int32)
            indices = np.stack((x, y, z), -1).reshape(-1, 3)
            self._centers = self.lo + (indices + .5) * self.step
        return self._centers

    def scene(self, scene_dir, frame_idx, ego_pose, intrinsics, extrinsics, image_shape):
        """仅用 PT 静态点生成体素；遮挡判定严格不引用 Agent 或 LiDAR。"""
        def build(payload):
            # 世界 AABB 是旋转后 BEV 盒的保守包络；体素索引仍由精确刚性变换决定。
            corners = np.stack(np.meshgrid(*zip(self.lo, self.hi), indexing="ij"), -1).reshape(-1, 3)
            world_corners = transform_points(corners, transform_matrix(ego_pose))
            minimum = world_corners.min(0) - 1e-9
            maximum = world_corners.max(0) + 1e-9
            if self.device.type == "cuda":
                scene = Path(scene_dir).name
                if self._gpu_static_scene != scene:
                    self._gpu_static_points = payload["static"]["xyz"].to(self.device)
                    self._gpu_static_scene = scene
                occupied = self._scene_voxels_gpu(
                    self._gpu_static_points, ego_pose, minimum, maximum)
            else:
                points = payload["static"]["xyz"].numpy()
                points = points[np.all((points >= minimum) & (points <= maximum), axis=1)]
                occupied = self._scene_voxels_cpu(points, ego_pose)
            mask = self._visibility(occupied, intrinsics, extrinsics, image_shape)
            return occupied, mask
        return self._read_or_build("scene", scene_dir, frame_idx, build)

    def agent(self, scene_dir, frame_idx, ego_pose, boxes, visibility_builder,
              scene_occupied, intrinsics, extrinsics, image_shape):
        """命中时复用可见性；未命中才判定运动框并栅格化完整 OBB。"""
        def build(_payload):
            visible = np.asarray(visibility_builder(), dtype=bool)
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
            return occupied, mask, visible
        return self._read_or_build("agent", scene_dir, frame_idx, build)

    def _scene_voxels_cpu(self, points, ego_pose):
        local = transform_points(points, world_to_ego(ego_pose))
        index = np.floor((local - self.lo) / self.step).astype(np.int64)
        valid = np.all((index >= 0) & (index < self.dims), axis=1)
        occupied = np.zeros(self.shape, dtype=bool)
        i = index[valid]
        occupied[i[:, 2], i[:, 0], i[:, 1]] = True
        return occupied

    def _scene_voxels_gpu(self, points, ego_pose, minimum=None, maximum=None):
        if isinstance(points, torch.Tensor) and minimum is not None:
            low_bound = torch.as_tensor(minimum - 1e-5, device=self.device,
                                        dtype=points.dtype)
            high_bound = torch.as_tensor(maximum + 1e-5, device=self.device,
                                         dtype=points.dtype)
            points = points[((points >= low_bound) & (points <= high_bound)).all(1)]
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

    def _camera_candidates(self, intrinsics, extrinsics, image_shape):
        """相同标定与图像大小只计算一次视场候选体素。"""
        key = (tuple(image_shape),
               tuple((item["fx"], item["fy"], item["cx"], item["cy"])
                     for item in intrinsics), np.asarray(extrinsics).tobytes())
        if key == self._view_key:
            return self._view_candidates
        centers = self._grid_centers()
        height, width = image_shape
        views = []
        for intrinsic, extrinsic in zip(intrinsics, extrinsics):
            matrix = transform_matrix(extrinsic)
            local = transform_points(centers, np.linalg.inv(matrix))
            depth = local[:, 0]
            safe = np.maximum(depth, np.finfo(float).eps)
            u = intrinsic["fx"] * local[:, 1] / safe + intrinsic["cx"]
            v = intrinsic["cy"] - intrinsic["fy"] * local[:, 2] / safe
            in_view = (depth > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
            views.append((np.flatnonzero(in_view), matrix[:3, 3]))
        self._view_key = key
        self._view_candidates = views
        self._ray_tables = None
        self._ray_gpu_tables = None
        return views

    def _configure_cpu_threads(self):
        if self._jit_pid != os.getpid():
            set_num_threads(min(self.cpu_threads, numba_config.NUMBA_NUM_THREADS))
            self._jit_pid = os.getpid()

    def _build_ray_tables(self, views):
        """按标定生成三相机 CSR 射线路径；超预算时保留精确 DDA 回退。"""
        if not self.ray_lookup_enabled or njit is None or self._ray_tables is False:
            return None
        if self._ray_tables is not None:
            return self._ray_tables
        self._configure_cpu_threads()
        centers = self._grid_centers()
        counts = [_ray_path_counts(origin, centers[ids], self.lo, self.step, self.dims)
                  for ids, origin in views]
        table_bytes = sum(ids.nbytes + (len(ids) + 1) * 8 + int(item.sum()) * 4
                          for (ids, _), item in zip(views, counts))
        peak_bytes = table_bytes * (2 if self.device.type == "cuda" else 1)
        if self.device.type == "cuda":
            peak_bytes += max(int(item.sum()) * 5 for item in counts)
        if peak_bytes > self.ray_lookup_max_bytes:
            self._ray_tables = False
            return None
        tables = []
        for (ids, origin), item in zip(views, counts):
            offsets = np.empty(len(ids) + 1, dtype=np.int64)
            offsets[0] = 0
            np.cumsum(item, out=offsets[1:])
            indices = np.empty(int(offsets[-1]), dtype=np.uint32)
            _ray_path_fill(origin, centers[ids], self.lo, self.step,
                           self.dims, offsets, indices)
            tables.append((ids, offsets, indices))
        self._ray_tables = tables
        return tables

    def _lookup_visibility_cpu(self, occupied, tables):
        flat = occupied.ravel()
        visible = np.zeros(flat.size, dtype=bool)
        for ids, offsets, indices in tables:
            visible[ids] |= _lookup_visible_cpu(flat, offsets, indices)
        return visible.reshape(self.shape)

    def _lookup_visibility_torch(self, occupied, tables):
        """GPU 一次 gather + 分段最大值规约每条射线，目标体素不在路径表中。"""
        if self._ray_gpu_tables is None:
            self._ray_gpu_tables = [(
                torch.as_tensor(ids, device=self.device),
                torch.as_tensor(np.diff(offsets), device=self.device),
                torch.as_tensor(indices.view(np.int32), device=self.device))
                for ids, offsets, indices in tables]
        flat = torch.as_tensor(occupied.ravel(), device=self.device)
        visible = torch.zeros(flat.numel(), dtype=torch.bool, device=self.device)
        for ids, lengths, indices in self._ray_gpu_tables:
            path_occupied = flat.index_select(0, indices).to(torch.float32)
            blocked = torch.segment_reduce(path_occupied, "max", lengths=lengths,
                                           initial=0)
            visible[ids] |= blocked == 0
        return visible.cpu().numpy().reshape(self.shape)

    def _visibility(self, occupied, intrinsics, extrinsics, image_shape):
        centers = self._grid_centers()
        views = self._camera_candidates(intrinsics, extrinsics, image_shape)
        tables = self._build_ray_tables(views)
        if tables is not None:
            return (self._lookup_visibility_torch(occupied, tables)
                    if self.device.type == "cuda" else self._lookup_visibility_cpu(
                        occupied, tables))
        visible = np.zeros(len(centers), dtype=bool)
        occupied_gpu = (torch.as_tensor(occupied, device=self.device)
                        if self.device.type == "cuda" else None)
        for in_view, origin in views:
            candidates = in_view[~visible[in_view]]
            batches = ([candidates] if occupied_gpu is None and njit is not None
                       else np.array_split(candidates, max(1, int(np.ceil(len(candidates) / self.chunk)))))
            for indices in batches:
                if len(indices):
                    visible[indices] = (self._unblocked_gpu(
                        occupied_gpu, origin, centers[indices])
                        if occupied_gpu is not None else self._unblocked(
                            occupied, origin, centers[indices]))
        return visible.reshape(self.shape)

    def _unblocked(self, occupied, origin, targets):
        """CPU 以多核 JIT 遍历独立射线；无 Numba 时保留原 NumPy 算子。"""
        if njit is not None:
            self._configure_cpu_threads()
            return _trace_rays_cpu(occupied, np.asarray(origin), targets,
                                   self.lo, self.step, self.dims)
        return self._unblocked_numpy(occupied, origin, targets)

    def _unblocked_numpy(self, occupied, origin, targets):
        """原始向量化 3D DDA，作为精确对照和无 JIT 环境回退。"""
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

