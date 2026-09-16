"""提供有容量上限、可并发回填的 BEVSeg 压缩 NumPy 磁盘缓存。

模块: data/bevseg_cache/bevseg_cache.py
依赖: numpy, torch, data.bevseg_cache.checks
读取配置: data.bevseg.cache.enabled/dir/max_size_gb/write_missing/compression_level
对外接口:
    - BevSegDiskCache(cfg, fingerprint_payload) -> object
      .load(key) -> dict[str, ndarray] | None
      .contains(key) -> bool
      .store(key, semantic, direction) -> str
      .summary() -> dict
    - prepare_bevseg_cache(dataset, num_workers, prefetch_factor, in_order, progress_every) -> dict
说明: 二值语义按 bit 打包；五帧共享的 float32 方向场只存一份，解码结果逐元素等价。
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import uuid
import zipfile

import numpy as np
from torch.utils.data import DataLoader, Dataset

from data.bevseg_cache.checks.bevseg_cache_checks import (
    check_cache_arrays,
    check_cache_payload,
    check_cache_prepare,
)


__all__ = ["BevSegDiskCache", "prepare_bevseg_cache"]


_FORMAT_VERSION = 1
_STATE_VERSION = 1


@contextmanager
def _exclusive_file_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class BevSegDiskCache:
    """以配置/实现指纹隔离条目，并在全局负载上限内原子写入。"""

    def __init__(self, cfg, fingerprint_payload, history_frames, layers, resolution) -> None:
        self.enabled = bool(cfg.enabled)
        self.writable = self.enabled and bool(cfg.write_missing)
        root = Path(cfg.dir)
        self.root = root if root.is_absolute() else Path(__file__).resolve().parents[2] / root
        self.max_size_bytes = int(float(cfg.max_size_gb) * 1024 ** 3)
        self.compression_level = int(cfg.compression_level)
        self.history_frames = int(history_frames)
        self.layers = int(layers)
        self.resolution = int(resolution)
        self.semantic_shape = (
            self.history_frames, self.layers, self.resolution, self.resolution)
        self.direction_shape = (2, self.resolution, self.resolution)
        fingerprint = {
            "format_version": _FORMAT_VERSION,
            "payload": fingerprint_payload,
        }
        encoded = json.dumps(fingerprint, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")).encode("utf-8")
        self.namespace_name = hashlib.sha256(encoded).hexdigest()[:16]
        self.namespace = self.root / self.namespace_name
        self._lock_path = self.root / ".cache.lock"
        self._state_path = self.root / ".usage.json"
        self._dirty_path = self.root / ".usage.dirty"
        if self.writable:
            self.namespace.mkdir(parents=True, exist_ok=True)
            with _exclusive_file_lock(self._lock_path):
                self._cleanup_temporary()
                self._write_manifest(fingerprint)
                usage = self._scan_usage()
                self._write_state(usage, sealed=usage >= self.max_size_bytes)
                self._dirty_path.unlink(missing_ok=True)

    def _path(self, key):
        digest = hashlib.sha256(str(key).encode("utf-8")).hexdigest()
        return self.namespace / digest[:2] / (digest + ".npz")

    def _scan_usage(self):
        if not self.root.exists():
            return 0
        return sum(path.stat().st_size for path in self.root.rglob("*.npz") if path.is_file())

    def _read_state(self):
        if self._dirty_path.exists():
            return {"size_bytes": self._scan_usage(), "sealed": False}
        try:
            state = json.loads(self._state_path.read_text(encoding="utf-8"))
            if state.get("version") != _STATE_VERSION or int(state["size_bytes"]) < 0:
                raise ValueError("invalid cache state")
            return {"size_bytes": int(state["size_bytes"]),
                    "sealed": bool(state.get("sealed", False))}
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return {"size_bytes": self._scan_usage(), "sealed": False}

    def _write_state(self, size_bytes, sealed):
        state = {"version": _STATE_VERSION, "size_bytes": int(size_bytes),
                 "sealed": bool(sealed)}
        self._atomic_write(self._state_path, json.dumps(state, sort_keys=True).encode("utf-8"))

    def _reconcile_state(self):
        state = self._read_state()
        if self._dirty_path.exists():
            state["sealed"] = state["size_bytes"] >= self.max_size_bytes
            self._write_state(state["size_bytes"], state["sealed"])
            self._dirty_path.unlink(missing_ok=True)
        return state

    def _write_manifest(self, fingerprint):
        path = self.namespace / "manifest.json"
        if path.exists():
            return
        content = json.dumps(fingerprint, ensure_ascii=False, indent=2,
                             sort_keys=True).encode("utf-8")
        self._atomic_write(path, content)

    def _cleanup_temporary(self):
        for path in self.root.rglob(".*.tmp"):
            if path.is_file():
                path.unlink()

    @staticmethod
    def _atomic_write(path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(".{}.{}.tmp".format(path.name, uuid.uuid4().hex))
        try:
            with temporary.open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _encode(self, semantic, direction):
        check_cache_arrays(
            semantic, direction, self.history_frames, self.layers, self.resolution)
        semantic_bits = np.packbits(semantic.reshape(-1).astype(np.uint8), bitorder="little")
        buffer = io.BytesIO()
        arrays = {"semantic_bits": semantic_bits,
                  "direction": np.ascontiguousarray(direction[0])}
        with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=self.compression_level) as archive:
            for name, array in arrays.items():
                with archive.open(name + ".npy", mode="w") as member:
                    np.lib.format.write_array(member, array, allow_pickle=False)
        return buffer.getvalue()

    def _decode(self, path):
        with np.load(path, allow_pickle=False) as payload:
            semantic_bits = payload["semantic_bits"]
            direction = payload["direction"]
        semantic_count = int(np.prod(self.semantic_shape))
        check_cache_payload(
            semantic_bits, direction, semantic_count, self.direction_shape)
        semantic = np.unpackbits(
            semantic_bits, count=semantic_count, bitorder="little").reshape(self.semantic_shape)
        semantic = np.ascontiguousarray(semantic, dtype=np.float32)
        direction = np.repeat(direction[None], self.history_frames, axis=0)
        return {"semantic": semantic, "direction": direction}

    def load(self, key):
        """读取并还原缓存栅格；未命中或负载损坏时返回 None。"""
        if not self.enabled:
            return None
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            return self._decode(path)
        except (OSError, EOFError, ValueError, KeyError, TypeError, zipfile.BadZipFile):
            self._discard(path)
            return None

    def contains(self, key):
        """仅检查条目文件是否存在，供启动时快速检测缺失缓存。"""
        return self.enabled and self._path(key).is_file()

    def _discard(self, path):
        if not self.writable:
            return
        with _exclusive_file_lock(self._lock_path):
            self._atomic_write(self._dirty_path, b"")
            if path.exists():
                path.unlink()
            actual = self._scan_usage()
            self._write_state(actual, sealed=False)
            self._dirty_path.unlink(missing_ok=True)

    def store(self, key, semantic, direction):
        """原子写入一个未命中条目，返回 written、hit、full 或 disabled。"""
        if not self.writable:
            return "disabled"
        path = self._path(key)
        if path.is_file():
            return "hit"
        content = self._encode(semantic, direction)
        with _exclusive_file_lock(self._lock_path):
            if path.is_file():
                return "hit"
            state = self._reconcile_state()
            projected = state["size_bytes"] + len(content)
            if state["sealed"] or projected > self.max_size_bytes:
                self._write_state(state["size_bytes"], sealed=True)
                return "full"
            self._atomic_write(self._dirty_path, b"")
            self._atomic_write(path, content)
            self._write_state(projected, sealed=False)
            self._dirty_path.unlink(missing_ok=True)
        return "written"

    def is_full(self):
        """返回缓存是否已因容量不足停止接收新条目。"""
        if not self.writable or not self._state_path.is_file():
            return False
        return self._read_state()["sealed"]

    def summary(self):
        """返回当前命名空间和全局缓存容量摘要。"""
        state = self._read_state() if self.root.exists() else {"size_bytes": 0, "sealed": False}
        return {
            "namespace": self.namespace_name,
            "size_bytes": state["size_bytes"],
            "max_size_bytes": self.max_size_bytes,
            "full": state["sealed"],
        }


class _CacheWarmupDataset(Dataset):
    def __init__(self, dataset, sample_indices) -> None:
        self.dataset = dataset
        self.sample_indices = sample_indices

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, position):
        sample_index = self.sample_indices[position]
        return {"sample_index": sample_index,
                "status": self.dataset.prepare_cache_sample(sample_index)}


def prepare_bevseg_cache(dataset, num_workers, prefetch_factor, in_order, progress_every):
    """自动检测并并行生成完整训练数据集中缺失的 BEVSeg 缓存。"""
    check_cache_prepare(dataset, num_workers, prefetch_factor, progress_every)
    total = len(dataset)
    missing = []
    for checked in range(1, total + 1):
        if not dataset.cache_contains(checked - 1):
            missing.append(checked - 1)
        if checked == 1 or checked % progress_every == 0 or checked == total:
            print("[bevseg-cache] scan {}/{} ({:.2f}%) missing={}".format(
                checked, total, checked * 100.0 / total, len(missing)), flush=True)

    existing = total - len(missing)
    print("[bevseg-cache] scan complete: existing={} pending={} train_samples={}".format(
        existing, len(missing), total), flush=True)
    if not missing:
        result = dataset.cache_summary()
        result.update(samples={"existing": existing}, complete=True)
        return result
    if dataset.cache_summary()["full"]:
        print("[bevseg-cache] capacity full; keeping existing entries and continuing.",
              flush=True)
        result = dataset.cache_summary()
        result.update(samples={"existing": existing, "full": 1}, complete=False)
        return result

    loader_kwargs = {
        "batch_size": None,
        "num_workers": int(num_workers),
        "persistent_workers": False,
    }
    if num_workers > 0:
        loader_kwargs.update(prefetch_factor=int(prefetch_factor), in_order=bool(in_order))
    loader = DataLoader(_CacheWarmupDataset(dataset, missing), **loader_kwargs)
    counts = Counter()
    processed = 0
    for processed, item in enumerate(loader, start=1):
        sample_index, status = int(item["sample_index"]), str(item["status"])
        counts[str(status)] += 1
        if (processed == 1 or processed % int(progress_every) == 0
                or processed == len(missing) or status == "full"):
            summary = dataset.cache_summary()
            print(
                "[bevseg-cache] dataset_sample={}/{} build={}/{} progress={:.2f}% "
                "size={:.3f}/{:.3f} GiB stats={}".format(
                    sample_index + 1, total, processed, len(missing),
                    processed * 100.0 / len(missing), summary["size_bytes"] / 1024 ** 3,
                    summary["max_size_bytes"] / 1024 ** 3, dict(counts)),
                flush=True,
            )
        if status == "full":
            break
    result = dataset.cache_summary()
    result["samples"] = {"existing": existing, **dict(counts)}
    result["complete"] = processed == len(missing) and counts.get("full", 0) == 0
    return result
