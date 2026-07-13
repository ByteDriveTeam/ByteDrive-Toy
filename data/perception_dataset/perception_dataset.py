"""感知模型单帧数据集：把落盘场景逐帧展开，产出归一化 RGB 与语义/深度监督目标（采用所有帧）。

模块: data/perception_dataset/perception_dataset.py
依赖: torch, numpy, config.schema.Config, data.single_frame_base.SingleFrameSceneBase, data.target_encoding
读取配置:
    data.dataset.scene_root / camera / dino_mean / dino_std
    model.physics.symlog_scale / depth_max_m
对外接口:
    - PerceptionDataset(cfg) -> torch.utils.data.Dataset
        __getitem__(i) -> dict[str, Tensor]   # rgb / semantic / depth_target / depth_inrange
        .frame_index -> list[(Path, int)]     # (场景目录, 帧号)，供可视化按场景筛选
说明: 单帧模型，每一帧独立成样本（采用所有帧，无开窗）。共享读取逻辑（索引/reader 缓存/RGB 归一化）下沉到
      data.single_frame_base.SingleFrameSceneBase（DRY，规范 §8），本类只负责取深度/语义并经 target_encoding
      在 Symlog 空间构建监督目标。RGB 保持原生分辨率不缩放（与 DINOv3 兼容）。
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from config.schema import Config
from data import target_encoding as te
from data.single_frame_base import SingleFrameSceneBase, writable_contiguous


__all__ = ["PerceptionDataset"]


class PerceptionDataset(SingleFrameSceneBase):
    """逐帧展开的单帧感知数据集。

    每个样本为一帧：
        rgb           [3, H, W]  归一化后
        semantic      [H, W]     long 语义标签
        depth_target  [H, W]     scale·symlog(depth)
        depth_inrange [H, W]     范围内掩码/二分类标签（1=范围内）
    """

    def __init__(self, cfg: Config) -> None:
        ds = cfg.data.dataset
        super().__init__(ds.scene_root, ds.camera, ds.dino_mean, ds.dino_std)
        self._physics = cfg.model.physics

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        scene_dir, frame_idx = self.frame_index[i]
        frame = self.reader(scene_dir).frame(frame_idx)

        rgb = self.normalize_rgb(frame["rgb"][self._camera])          # [3,H,W]
        depth = _to_float(frame["depth"][self._camera])               # [H,W]
        semantic = _to_long(frame["semantic"][self._camera])          # [H,W]

        depth_target, depth_inrange = te.depth_targets(
            depth, self._physics.symlog_scale, self._physics.depth_max_m)

        return {"rgb": rgb, "semantic": semantic,
                "depth_target": depth_target, "depth_inrange": depth_inrange}


def _to_float(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(writable_contiguous(arr)).float()


def _to_long(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(writable_contiguous(arr)).long()
