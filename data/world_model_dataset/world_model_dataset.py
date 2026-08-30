"""按场景边界直接读取位压缩 LMDB，并组装 5 帧 10Hz 世界模型输入。

模块: data/world_model_dataset/world_model_dataset.py
依赖: bisect, collections, lmdb, msgpack, numpy, torch, zlib, config.schema.Config, 本模块 checks
读取配置:
    data.world_model_dataset.root / history_stride / scene_cache_size
    model.world_model.num_frames / grid.front_m / rear_m / left_m / right_m /
        cell_size_m / layer_names
对外接口:
    - WorldModelDataset(cfg) -> Dataset
    - decode_grid(blob, shape) -> ndarray
说明: 索引仅保存逐场景累计长度而不展开百万级窗口元组；每个 DataLoader worker 使用有界
      LMDB 句柄 LRU。解码只做 zlib + unpackbits，训练端得到紧凑 uint8 后再搬运到设备。
"""

from __future__ import annotations

import bisect
import zlib
from collections import OrderedDict
from pathlib import Path

import lmdb
import msgpack
import numpy as np
import torch
from torch.utils.data import Dataset

from config.schema import Config
from data.world_model_dataset.checks.world_model_dataset_checks import (
    check_dataset_root,
    check_scene_meta,
)


__all__ = ["WorldModelDataset", "decode_grid"]


def decode_grid(blob: bytes, shape) -> np.ndarray:
    """解码 packbits_little_zlib 二值栅格为 uint8 ndarray。"""
    count = int(np.prod(shape))
    packed = np.frombuffer(zlib.decompress(blob), dtype=np.uint8)
    return np.unpackbits(packed, bitorder="little", count=count).reshape(shape)


class WorldModelDataset(Dataset):
    """返回同场景内时序对齐的 `[T,10,H,W]` uint8 栅格窗口。"""

    def __init__(self, cfg: Config) -> None:
        data_cfg = cfg.data.world_model_dataset
        self._root = _repo_path(data_cfg.root)
        check_dataset_root(self._root)
        self._frames = int(cfg.model.world_model.num_frames)
        self._stride = int(data_cfg.history_stride)
        self._cache_size = int(data_cfg.scene_cache_size)
        grid = cfg.model.world_model.grid
        height = int(round((grid.front_m + grid.rear_m) / grid.cell_size_m))
        width = int(round((grid.left_m + grid.right_m) / grid.cell_size_m))
        self._shape = (len(grid.layer_names), height, width)
        self._layers = tuple(grid.layer_names)
        self._scenes, counts = self._scan_scenes()
        self._cumulative = np.cumsum(counts, dtype=np.int64).tolist()
        self._handles = OrderedDict()

    def __len__(self) -> int:
        return self._cumulative[-1] if self._cumulative else 0

    def __getitem__(self, index: int):
        scene_index, local_index = self._locate(index)
        env, meta = self._reader(scene_index)
        first_end = (self._frames - 1) * self._stride
        end = first_end + local_index
        indices = np.arange(end - first_end, end + 1, self._stride, dtype=np.int64)
        with env.begin() as txn:
            grids = [decode_grid(txn.get("grid/{:08d}".format(int(i)).encode()), self._shape)
                     for i in indices]
        return {
            "grid": torch.from_numpy(np.stack(grids)),
            "scene_index": torch.tensor(scene_index, dtype=torch.int64),
            "frame_index": torch.tensor(end, dtype=torch.int64),
        }

    def close(self) -> None:
        """关闭当前进程内缓存的全部 LMDB 句柄。"""
        handles = getattr(self, "_handles", {})
        for env, _ in handles.values():
            env.close()
        handles.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = OrderedDict()
        return state

    def __del__(self):
        self.close()

    def _scan_scenes(self):
        scenes, counts = [], []
        history = (self._frames - 1) * self._stride
        for scene in sorted(path for path in self._root.glob("scene_*") if path.is_dir()):
            env = lmdb.open(str(scene / "lmdb"), readonly=True, subdir=True, lock=False)
            try:
                with env.begin() as txn:
                    meta = msgpack.unpackb(txn.get(b"meta"), raw=False)
                check_scene_meta(meta, self._shape, self._layers)
                count = int(meta["num_frames"]) - history
                if count > 0:
                    scenes.append(scene)
                    counts.append(count)
            finally:
                env.close()
        if not scenes:
            raise RuntimeError("世界模型栅格目录没有足够长的完整场景：{}".format(self._root))
        return scenes, counts

    def _locate(self, index):
        length = len(self)
        normalized = index + length if index < 0 else index
        if normalized < 0 or normalized >= length:
            raise IndexError("世界模型样本索引越界：{} / {}".format(index, length))
        scene_index = bisect.bisect_right(self._cumulative, normalized)
        previous = self._cumulative[scene_index - 1] if scene_index else 0
        return scene_index, normalized - previous

    def _reader(self, scene_index):
        if scene_index in self._handles:
            self._handles.move_to_end(scene_index)
            return self._handles[scene_index]
        scene = self._scenes[scene_index]
        env = lmdb.open(str(scene / "lmdb"), readonly=True, subdir=True, lock=False,
                        readahead=False, max_readers=256)
        with env.begin() as txn:
            meta = msgpack.unpackb(txn.get(b"meta"), raw=False)
        self._handles[scene_index] = (env, meta)
        if len(self._handles) > self._cache_size:
            _, (old_env, _) = self._handles.popitem(last=False)
            old_env.close()
        return env, meta


def _repo_path(value):
    path = Path(value)
    return path if path.is_absolute() else Path(__file__).resolve().parents[2] / path
