"""场景感知批采样器：连续帧同批、批间随机，减少视频随机 seek 与跨场景解码器切换。

模块: data/scene_batch_sampler/scene_batch_sampler.py
依赖: torch, torch.utils.data.Sampler
读取配置: —（batch_size/shuffle/drop_last 由调用方传入，来源 config.train.*）
对外接口:
    - SceneBatchSampler(frame_index, batch_size, shuffle, drop_last) -> Sampler[list[int]]
说明: 每个场景先切出连续完整批，再在批粒度打乱；各场景不足一批的尾帧集中补批，因而 drop_last
      只影响全数据集最后不足一批的样本。连续帧让 H.265 解码走顺序 read，同时保持训练 step 的批间随机性。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterator, List, Sequence, Tuple

import torch
from torch.utils.data import Sampler


__all__ = ["SceneBatchSampler"]


class SceneBatchSampler(Sampler[List[int]]):
    """把同场景连续帧组成 batch，并按 batch 粒度打乱顺序。"""

    def __init__(self, frame_index: Sequence[Tuple[object, int]], batch_size: int,
                 shuffle: bool, drop_last: bool) -> None:
        scene_indices = defaultdict(list)
        for index, (scene, _) in enumerate(frame_index):
            scene_indices[scene].append(index)
        self._scenes = tuple(tuple(indices) for indices in scene_indices.values())
        self._num_samples = len(frame_index)
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        batches, tails = [], []
        for indices in self._scenes:
            full_end = len(indices) - len(indices) % self._batch_size
            batches.extend(list(indices[start:start + self._batch_size])
                           for start in range(0, full_end, self._batch_size))
            tails.extend(indices[full_end:])

        if self._shuffle and tails:
            tails = [tails[i] for i in torch.randperm(len(tails)).tolist()]
        batches.extend(tails[start:start + self._batch_size]
                       for start in range(0, len(tails), self._batch_size)
                       if not self._drop_last or start + self._batch_size <= len(tails))
        order = torch.randperm(len(batches)).tolist() if self._shuffle else range(len(batches))
        return iter(batches[i] for i in order)

    def __len__(self) -> int:
        if self._drop_last:
            return self._num_samples // self._batch_size
        return (self._num_samples + self._batch_size - 1) // self._batch_size
