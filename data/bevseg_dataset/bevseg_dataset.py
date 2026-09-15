"""读取 CARLA 场景并返回五帧 ego 对齐 BEVSeg。

模块: data/bevseg_dataset/bevseg_dataset.py
依赖: torch, numpy, data.single_frame_base, data.bevseg_synthesis
读取配置: data.bevseg.scene_root/history_frames/window_stride_s, data.scene_cache_size
对外接口:
    - BevSegDataset(cfg) -> Dataset
      __getitem__(index) -> dict[str, Tensor]
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

from data.bevseg_synthesis import BevSegRasterizer
from data.single_frame_base import resolve_repo_path
from vis.data_vis.reader import SceneReader, list_scenes


__all__ = ["BevSegDataset"]


class BevSegDataset(Dataset):
    """返回当前帧坐标系下的完整五帧 BEVSeg 样本。"""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.data_cfg = cfg.data.bevseg
        self._root = resolve_repo_path(self.data_cfg.scene_root)
        self._rasterizer = BevSegRasterizer(self.data_cfg)
        self._cache_size = cfg.data.scene_cache_size
        self._readers = OrderedDict()
        self._maps = OrderedDict()
        self._index = self._build_index()

    def __len__(self):
        return len(self._index)

    @property
    def frame_index(self):
        """返回 (scene_dir, frame_idx) 索引。"""
        return self._index

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
        history = self.data_cfg.history_frames
        index = []
        for scene in list_scenes(self._root):
            reader = SceneReader(scene)
            try:
                # failed 只描述驾驶结果，不代表 BEV 标签损坏；压缩学习保留所有完整窗口。
                if reader.num_frames < history:
                    continue
                # 不同场景可采用不同落盘频率，按场景元数据保持统一的时间步长。
                sensor_dt_s = float(reader.meta["sensor_dt_s"])
                stride = max(1, int(round(self.data_cfg.window_stride_s / sensor_dt_s)))
                index.extend((scene, frame)
                             for frame in range(history - 1, reader.num_frames, stride))
            finally:
                reader.close()
        return index

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

    def _map(self, scene, scene_meta):
        key = str(scene)
        map_obj = self._maps.pop(key, None)
        if map_obj is None:
            map_obj = self._rasterizer.load_map(scene_meta)
        self._maps[key] = map_obj
        if len(self._maps) > self._cache_size:
            self._maps.popitem(last=False)
        return map_obj

    def __getitem__(self, index):
        scene, frame_idx = self._index[index]
        reader = self._reader(scene)
        current_meta = reader.frame_meta(frame_idx)
        current_pose = current_meta["ego"]["transform"]
        scene_meta = reader.meta
        map_obj = self._map(scene, scene_meta)
        start = frame_idx - self.data_cfg.history_frames + 1
        outputs = [
            self._rasterizer.rasterize_frame(
                scene_meta, reader.frame_meta(i), map_obj, current_pose)
            for i in range(start, frame_idx + 1)
        ]
        semantic = np.stack([item["semantic"] for item in outputs], axis=0)
        direction = np.stack([item["direction"] for item in outputs], axis=0)
        bevseg = np.concatenate((semantic, direction), axis=1)
        return {
            "bevseg": torch.from_numpy(np.ascontiguousarray(bevseg)).float(),
            "semantic": torch.from_numpy(np.ascontiguousarray(semantic)).float(),
            "direction": torch.from_numpy(np.ascontiguousarray(direction)).float(),
            "temporal_valid": torch.ones(self.data_cfg.history_frames, dtype=torch.float32),
        }
