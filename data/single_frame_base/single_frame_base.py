"""单帧场景数据集共享基类：场景/帧索引、有界 SceneReader 缓存、RGB 归一化（感知与驾驶数据集复用）。

模块: data/single_frame_base/single_frame_base.py
依赖: torch, numpy, lmdb, msgpack, vis.data_vis.reader(SceneReader/list_scenes),
      data.single_frame_base.checks.single_frame_base_checks
读取配置: —（scene_root/camera/dino_mean/dino_std/scene_cache_size 由子类以参数传入，来源 config.data.*）
对外接口:
    - SingleFrameSceneBase(scene_root, camera, dino_mean, dino_std, scene_cache_size) -> Dataset
        .frame_index -> list[(Path, int)]       # (场景目录, 帧号)
        .reader(scene_dir) -> SceneReader        # 惰性构造并按 worker 缓存
        .normalize_rgb(bgr) -> Tensor            # BGR uint8 → DINO 归一化 [3,H,W]
        .scene_num_frames(scene_dir) -> int      # 轻量读 LMDB num_frames
说明: 把感知/驾驶两个单帧数据集的公共读取逻辑收拢一处（DRY，规范 §8）：索引期只用 LMDB 轻量读 num_frames、
      不建 VideoCapture；RGB 由 BGR→RGB、/255、DINO ImageNet 归一化并保持原生分辨率（与 DINOv3 兼容）；
      SceneReader 惰性按需构造并在每个 worker 内做有界 LRU，淘汰时显式关闭视频与 LMDB，避免场景数增长导致 OOM。
      cv2.VideoCapture 非跨进程安全，故缓存不会跨 worker 共享。子类只需实现 __getitem__。
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple

import lmdb
import msgpack
import numpy as np
import torch
from torch.utils.data import Dataset

from data.single_frame_base.checks.single_frame_base_checks import check_has_frames, check_scene_root
from vis.data_vis.reader import SceneReader, list_scenes


__all__ = ["SingleFrameSceneBase", "resolve_repo_path"]


class SingleFrameSceneBase(Dataset):
    """单帧场景数据集基类：共享索引、有界 reader 缓存与 RGB 归一化。"""

    def __init__(self, scene_root, camera: str, dino_mean, dino_std,
                 scene_cache_size: int) -> None:
        self._root = resolve_repo_path(scene_root)
        check_scene_root(self._root)
        self._camera = camera
        self._mean = torch.tensor(dino_mean, dtype=torch.float32).view(3, 1, 1)
        self._std = torch.tensor(dino_std, dtype=torch.float32).view(3, 1, 1)
        self._index = _build_frame_index(self._root)
        check_has_frames(self._index, self._root)
        self._scene_cache_size = scene_cache_size
        self._readers = OrderedDict()

    def __len__(self) -> int:
        return len(self._index)

    @property
    def frame_index(self) -> List[Tuple[Path, int]]:
        """只读暴露 (场景目录, 帧号) 列表，供可视化/采样按场景筛选帧。"""
        return self._index

    def reader(self, scene_dir: Path) -> SceneReader:
        """返回场景 reader；每 worker 采用有界 LRU，淘汰时立即释放原生解码与映射资源。"""
        key = str(scene_dir)
        reader = self._readers.pop(key, None)
        if reader is None:
            reader = SceneReader(scene_dir)
        self._readers[key] = reader
        if len(self._readers) > self._scene_cache_size:
            _, evicted = self._readers.popitem(last=False)
            evicted.close()
        return reader

    def normalize_rgb(self, bgr: np.ndarray) -> torch.Tensor:
        """BGR uint8 → RGB → [0,1] → DINO ImageNet 归一化，输出 [3,H,W] float32。"""
        rgb = torch.from_numpy(np.ascontiguousarray(bgr[:, :, ::-1])).float() / 255.0
        rgb = rgb.permute(2, 0, 1)
        return (rgb - self._mean) / self._std

    @staticmethod
    def scene_num_frames(scene_dir: Path) -> int:
        """轻量读场景 LMDB 的 num_frames（不建 VideoCapture），场景不可读则返回 0。"""
        return _scene_num_frames(scene_dir)


def resolve_repo_path(path) -> Path:
    """把相对路径解析到仓库根下（本文件位于 <root>/data/single_frame_base/）。"""
    p = Path(path)
    return p if p.is_absolute() else Path(__file__).resolve().parents[2] / p


def writable_contiguous(arr: np.ndarray) -> np.ndarray:
    """确保 C 连续且可写：只读数组（如 np.frombuffer 读 LMDB）会被拷贝，避免 torch.from_numpy 警告。"""
    arr = np.ascontiguousarray(arr)
    return arr if arr.flags.writeable else arr.copy()


def _scene_num_frames(scene_dir: Path) -> int:
    """轻量读场景 LMDB 的 num_frames（不建 VideoCapture），场景不可读则返回 0。"""
    try:
        env = lmdb.open(str(scene_dir / "lmdb"), readonly=True, subdir=True, lock=False)
    except lmdb.Error:
        return 0
    try:
        with env.begin() as txn:
            blob = txn.get(b"num_frames")
        return int(msgpack.unpackb(blob)) if blob is not None else 0
    finally:
        env.close()


def _build_frame_index(root: Path) -> List[Tuple[Path, int]]:
    """遍历场景，枚举每一帧，构建 (场景目录, 帧号) 索引（采用所有帧）。"""
    return [(scene_dir, frame_idx)
            for scene_dir in list_scenes(root)
            for frame_idx in range(_scene_num_frames(scene_dir))]
