"""感知模型单帧数据集：把落盘场景逐帧展开，产出归一化 RGB 与语义/深度监督目标（采用所有帧）。

模块: data/perception_dataset/perception_dataset.py
依赖: torch, numpy, lmdb, msgpack, vis.data_vis.reader.SceneReader, data.target_encoding,
      config.schema.Config, data.perception_dataset.checks.perception_dataset_checks
读取配置:
    data.dataset.scene_root / camera / dino_mean / dino_std
    model.physics.symlog_scale / depth_max_m
对外接口:
    - PerceptionDataset(cfg) -> torch.utils.data.Dataset
        __getitem__(i) -> dict[str, Tensor]   # rgb / semantic / depth_target / depth_inrange
        .frame_index -> list[(Path, int)]     # (场景目录, 帧号)，供可视化按场景筛选
说明: 单帧模型，每一帧独立成样本（采用所有帧，无开窗）。复用 vis.data_vis.reader.SceneReader（DRY，共用
      unpack_array 与 LMDB 键逻辑）读取逐帧数据；索引期仅用 LMDB 轻量读 num_frames，避免为每个场景预建
      VideoCapture。RGB 由 BGR→RGB、/255、DINO ImageNet 归一化，保持原生分辨率不缩放（与 DINOv3 兼容）。
      深度目标经 data.target_encoding 在 Symlog 空间构建。SceneReader 惰性按需构造并按 worker 缓存
      （cv2.VideoCapture 非跨进程安全）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import lmdb
import msgpack
import numpy as np
import torch
from torch.utils.data import Dataset

from config.schema import Config
from data import target_encoding as te
from data.perception_dataset.checks.perception_dataset_checks import check_scene_root, check_has_frames
from vis.data_vis.reader import SceneReader, list_scenes


__all__ = ["PerceptionDataset"]


class PerceptionDataset(Dataset):
    """逐帧展开的单帧感知数据集。

    Args:
        cfg: 全局配置，读取 `data.dataset` 与 `model.physics`。

    每个样本为一帧：
        rgb           [3, H, W]  归一化后
        semantic      [H, W]     long 语义标签
        depth_target  [H, W]     scale·symlog(depth)
        depth_inrange [H, W]     范围内掩码/二分类标签（1=范围内）
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        ds = cfg.data.dataset
        self._root = _resolve(ds.scene_root)
        check_scene_root(self._root)
        self._camera = ds.camera
        self._mean = torch.tensor(ds.dino_mean, dtype=torch.float32).view(3, 1, 1)
        self._std = torch.tensor(ds.dino_std, dtype=torch.float32).view(3, 1, 1)
        self._physics = cfg.model.physics
        # (场景目录, 帧号) 列表；索引期只读 num_frames，不建 VideoCapture
        self._index = _build_frame_index(self._root)
        check_has_frames(self._index, self._root)
        self._readers: Dict[str, SceneReader] = {}  # 按 worker 惰性缓存

    def __len__(self) -> int:
        return len(self._index)

    @property
    def frame_index(self) -> List[Tuple[Path, int]]:
        """只读暴露 (场景目录, 帧号) 列表，供可视化按场景筛选帧。"""
        return self._index

    def _reader(self, scene_dir: Path) -> SceneReader:
        """惰性构造并缓存该场景的 SceneReader（每 worker 一份，避免跨进程共享解码器）。"""
        key = str(scene_dir)
        if key not in self._readers:
            self._readers[key] = SceneReader(scene_dir)
        return self._readers[key]

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        scene_dir, frame_idx = self._index[i]
        frame = self._reader(scene_dir).frame(frame_idx)

        rgb = self._normalize_rgb(frame["rgb"][self._camera])         # [3,H,W]
        depth = _to_float(frame["depth"][self._camera])               # [H,W]
        semantic = _to_long(frame["semantic"][self._camera])          # [H,W]

        depth_target, depth_inrange = te.depth_targets(
            depth, self._physics.symlog_scale, self._physics.depth_max_m)

        return {"rgb": rgb, "semantic": semantic,
                "depth_target": depth_target, "depth_inrange": depth_inrange}

    def _normalize_rgb(self, bgr: np.ndarray) -> torch.Tensor:
        """BGR uint8 → RGB → [0,1] → DINO ImageNet 归一化，输出 [3,H,W] float32。"""
        rgb = torch.from_numpy(np.ascontiguousarray(bgr[:, :, ::-1])).float() / 255.0
        rgb = rgb.permute(2, 0, 1)  # [3,H,W]
        return (rgb - self._mean) / self._std


def _resolve(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else Path(__file__).resolve().parents[2] / p


def _writable_contiguous(arr: np.ndarray) -> np.ndarray:
    """确保 C 连续且可写：只读数组（如 np.frombuffer 读 LMDB）会被拷贝，避免 torch.from_numpy 警告。"""
    arr = np.ascontiguousarray(arr)
    return arr if arr.flags.writeable else arr.copy()


def _to_float(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(_writable_contiguous(arr)).float()


def _to_long(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(_writable_contiguous(arr)).long()


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
