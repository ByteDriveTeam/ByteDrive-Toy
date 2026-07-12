"""感知模型时序开窗数据集：把落盘场景切成 5 帧窗口，产出归一化 RGB 与四任务监督目标。

模块: data/perception_dataset.py
依赖: torch, numpy, lmdb, msgpack, vis.data_vis.reader.SceneReader, data.target_encoding,
      config.schema.Config, data.perception_dataset_checks
读取配置:
    data.dataset.scene_root / camera / window_size / window_stride / dino_mean / dino_std
    model.physics.symlog_scale / depth_max_m / flow_dt_s / flow_ndc_pixel_scale
对外接口:
    - PerceptionDataset(cfg) -> torch.utils.data.Dataset
        __getitem__(i) -> dict[str, Tensor]   # rgb / semantic / depth_target / depth_inrange / flow_target
        .window_index -> list[(Path, int)]    # (场景目录, 窗口起始帧)，供可视化按场景筛选
说明: 复用 vis.data_vis.reader.SceneReader（DRY，共用 unpack_array 与 LMDB 键逻辑）读取逐帧数据；
      索引期仅用 LMDB 轻量读 num_frames，避免为每个场景预建 VideoCapture。窗口在场景内连续取 window_size
      帧、步长 window_stride（=size 即不重叠）。RGB 由 BGR→RGB、/255、DINO ImageNet 归一化，保持原生分辨率
      不缩放（与 DINOv3 兼容）。深度/光流目标经 data.target_encoding 在 Symlog 空间构建；光流借当前帧深度
      与场景内参换算为图像平面速度。SceneReader 惰性按需构造并按 worker 缓存（cv2.VideoCapture 非跨进程安全）。
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
from data.perception_dataset_checks import check_scene_root, check_window_fits
from vis.data_vis.reader import SceneReader, list_scenes


__all__ = ["PerceptionDataset"]


class PerceptionDataset(Dataset):
    """按场景开窗的时序感知数据集。

    Args:
        cfg: 全局配置，读取 `data.dataset` 与 `model.physics`。

    每个样本为一个 window：
        rgb           [T, 3, H, W]  归一化后
        semantic      [T, H, W]     long 语义标签
        depth_target  [T, H, W]     scale·symlog(depth)
        depth_inrange [T, H, W]     范围内掩码/二分类标签（1=范围内）
        flow_target   [T, 2, H, W]  scale·symlog(图像平面速度)
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        ds = cfg.data.dataset
        self._root = _resolve(ds.scene_root)
        check_scene_root(self._root)
        self._camera = ds.camera
        self._window = ds.window_size
        self._stride = ds.window_stride
        self._mean = torch.tensor(ds.dino_mean, dtype=torch.float32).view(3, 1, 1)
        self._std = torch.tensor(ds.dino_std, dtype=torch.float32).view(3, 1, 1)
        self._physics = cfg.model.physics
        # (场景目录, 窗口起始帧) 列表；索引期只读 num_frames，不建 VideoCapture
        self._index = _build_window_index(self._root, self._window, self._stride)
        check_window_fits(self._index, self._root)
        self._readers: Dict[str, SceneReader] = {}  # 按 worker 惰性缓存

    def __len__(self) -> int:
        return len(self._index)

    @property
    def window_index(self) -> List[Tuple[Path, int]]:
        """只读暴露 (场景目录, 窗口起始帧) 列表，供可视化按场景筛选窗口。"""
        return self._index

    def _reader(self, scene_dir: Path) -> SceneReader:
        """惰性构造并缓存该场景的 SceneReader（每 worker 一份，避免跨进程共享解码器）。"""
        key = str(scene_dir)
        if key not in self._readers:
            self._readers[key] = SceneReader(scene_dir)
        return self._readers[key]

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        scene_dir, start = self._index[i]
        reader = self._reader(scene_dir)
        intr = reader.meta["intrinsics"][self._camera]
        frames = [reader.frame(start + t) for t in range(self._window)]

        rgb = torch.stack([self._normalize_rgb(f["rgb"][self._camera]) for f in frames])  # [T,3,H,W]
        depth = torch.stack([_to_float(f["depth"][self._camera]) for f in frames])         # [T,H,W]
        semantic = torch.stack([_to_long(f["semantic"][self._camera]) for f in frames])    # [T,H,W]
        flow = torch.stack([_to_float(f["optical_flow"][self._camera]) for f in frames])   # [T,H,W,2]

        depth_target, depth_inrange = te.depth_targets(
            depth, self._physics.symlog_scale, self._physics.depth_max_m)
        flow_target = te.flow_velocity_targets(
            flow, depth, float(intr["fx"]), float(intr["fy"]),
            self._physics.flow_ndc_pixel_scale, self._physics.flow_dt_s, self._physics.symlog_scale)
        # target_encoding 产出 [T,2,H,W]；对齐模型输出的 [C,T,H,W] 布局（batch 后 [B,2,T,H,W]）
        flow_target = flow_target.permute(1, 0, 2, 3).contiguous()

        return {"rgb": rgb, "semantic": semantic, "depth_target": depth_target,
                "depth_inrange": depth_inrange, "flow_target": flow_target}

    def _normalize_rgb(self, bgr: np.ndarray) -> torch.Tensor:
        """BGR uint8 → RGB → [0,1] → DINO ImageNet 归一化，输出 [3,H,W] float32。"""
        rgb = torch.from_numpy(np.ascontiguousarray(bgr[:, :, ::-1])).float() / 255.0
        rgb = rgb.permute(2, 0, 1)  # [3,H,W]
        return (rgb - self._mean) / self._std


def _resolve(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else Path(__file__).resolve().parents[1] / p


def _to_float(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr)).float()


def _to_long(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr)).long()


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


def _build_window_index(root: Path, window: int, stride: int) -> List[Tuple[Path, int]]:
    """遍历场景，按 window/stride 枚举每个窗口起始帧，构建 (场景目录, 起始帧) 索引。"""
    index: List[Tuple[Path, int]] = []
    for scene_dir in list_scenes(root):
        num_frames = _scene_num_frames(scene_dir)
        # range 上界 num_frames-window+1；不足一个窗口的场景自然跳过
        index.extend((scene_dir, start) for start in range(0, num_frames - window + 1, stride))
    return index
