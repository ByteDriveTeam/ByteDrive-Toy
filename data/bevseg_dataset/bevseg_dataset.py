"""读取 CARLA 场景并返回单帧 ego 对齐 BEVSeg。

模块: data/bevseg_dataset/bevseg_dataset.py
依赖: torch, numpy, data.single_frame_base, data.bevseg_synthesis, data.bevseg_cache
读取配置: data.bevseg.scene_root/sample_interval_s/cache, data.scene_cache_size
对外接口:
    - BevSegDataset(cfg) -> Dataset
      __getitem__(index) -> dict[str, Tensor]
      .cache_contains(index) -> bool
      .prepare_cache_sample(index) -> str
      .cache_summary() -> dict
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data.bevseg_cache import BevSegDiskCache
from data.bevseg_synthesis import BevSegRasterizer
from data.single_frame_base import resolve_repo_path
from vis.data_vis.reader import SceneReader, list_scenes


__all__ = ["BevSegDataset"]


class BevSegDataset(Dataset):
    """返回当前帧坐标系下的单帧 BEVSeg 样本。"""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.data_cfg = cfg.data.bevseg
        self._root = resolve_repo_path(self.data_cfg.scene_root)
        self._rasterizer = BevSegRasterizer(self.data_cfg)
        self._cache_size = cfg.data.scene_cache_size
        self._readers = OrderedDict()
        # 地图数量由有限 Town 集合约束；跨场景常驻可避免反复解压和构建空间索引。
        self._maps = {}
        self._scene_signatures = {}
        rasterizer_source = Path(self._rasterizer.__class__.__module__.replace(".", "/") + ".py")
        rasterizer_source = Path(__file__).resolve().parents[2] / rasterizer_source
        data_fingerprint = asdict(self.data_cfg)
        data_fingerprint.pop("cache")
        fingerprint = {
            "data": data_fingerprint,
            "rasterizer_sha256": hashlib.sha256(rasterizer_source.read_bytes()).hexdigest(),
        }
        self._disk_cache = BevSegDiskCache(
            self.data_cfg.cache,
            fingerprint,
            1,
            len(self.data_cfg.layers),
            self.data_cfg.resolution,
        )
        self._index = self._build_index()

    def __len__(self):
        return len(self._index)

    @property
    def frame_index(self):
        """返回 (scene_dir, frame_idx) 索引。"""
        return self._index

    @property
    def cache_enabled(self):
        """返回训练数据缓存是否启用。"""
        return self._disk_cache.enabled

    @property
    def cache_writable(self):
        """返回训练数据缓存是否允许回填。"""
        return self._disk_cache.writable

    def close(self):
        """关闭 LRU 中的场景读取器并释放场景缓存。"""
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()
        self._maps.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _build_index(self):
        index = []
        for scene in list_scenes(self._root):
            reader = SceneReader(scene)
            try:
                self._scene_signatures[str(scene)] = self._source_signature(scene, reader.meta)
                # failed 只描述驾驶结果，不代表 BEV 标签损坏；压缩学习保留所有完整窗口。
                if reader.num_frames < 1:
                    continue
                # 不同场景可采用不同落盘频率，按场景元数据保持统一的时间步长。
                sensor_dt_s = float(reader.meta["sensor_dt_s"])
                stride = max(1, int(round(self.data_cfg.sample_interval_s / sensor_dt_s)))
                index.extend((scene, frame)
                             for frame in range(0, reader.num_frames, stride))
            finally:
                reader.close()
        return index

    @staticmethod
    def _file_stamp(path):
        path = Path(path)
        if not path.is_file():
            return {"path": str(path.resolve()), "missing": True}
        stat = path.stat()
        return {"path": str(path.resolve()), "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns}

    def _source_signature(self, scene, scene_meta):
        payload = {
            "scene_lmdb": self._file_stamp(Path(scene) / "lmdb" / "data.mdb"),
            "hd_map": self._file_stamp(self._rasterizer.map_path(scene_meta)),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _cache_key(self, scene, frame_idx):
        return "{}:{}:{}".format(
            Path(scene).name, self._scene_signatures[str(scene)], int(frame_idx))

    def _reader(self, scene):
        key = str(scene)
        reader = self._readers.pop(key, None)
        if reader is None:
            reader = SceneReader(scene)
        self._readers[key] = reader
        if len(self._readers) > self._cache_size:
            _, evicted = self._readers.popitem(last=False)
            evicted.close()
        return reader

    def _map(self, scene_meta):
        key = str(self._rasterizer.map_path(scene_meta).resolve())
        map_obj = self._maps.get(key)
        if map_obj is None:
            map_obj = self._rasterizer.load_map(scene_meta)
        self._maps[key] = map_obj
        return map_obj

    def _rasterize(self, index):
        scene, frame_idx = self._index[index]
        reader = self._reader(scene)
        current_meta = reader.frame_meta(frame_idx)
        current_pose = current_meta["ego"]["transform"]
        scene_meta = reader.meta
        map_obj = self._map(scene_meta)
        return self._rasterizer.rasterize_frame(
            scene_meta, current_meta, map_obj, current_pose)

    def _load_outputs(self, index):
        scene, frame_idx = self._index[index]
        key = self._cache_key(scene, frame_idx)
        outputs = self._disk_cache.load(key)
        if outputs is not None:
            return outputs, "hit"
        outputs = self._rasterize(index)
        return outputs, self._disk_cache.store(
            key, outputs["semantic"], outputs["direction"])

    def prepare_cache_sample(self, index):
        """生成一个缓存条目，供离线并行预热使用。"""
        if self._disk_cache.is_full():
            return "full"
        _, status = self._load_outputs(index)
        return status

    def cache_contains(self, index):
        """快速判断一个训练样本是否已有缓存文件。"""
        scene, frame_idx = self._index[index]
        return self._disk_cache.contains(self._cache_key(scene, frame_idx))

    def cache_summary(self):
        """返回磁盘缓存容量摘要。"""
        return self._disk_cache.summary()

    def __getitem__(self, index):
        outputs, _ = self._load_outputs(index)
        semantic = outputs["semantic"]
        direction = outputs["direction"]
        bevseg = np.concatenate((semantic, direction), axis=0)
        return {
            "bevseg": torch.from_numpy(np.ascontiguousarray(bevseg)).float(),
            "semantic": torch.from_numpy(np.ascontiguousarray(semantic)).float(),
            "direction": torch.from_numpy(np.ascontiguousarray(direction)).float(),
        }
