"""场景感知批采样器：连续帧同批、批间随机。公开 API 重导出入口。

模块: data/scene_batch_sampler/__init__.py
依赖: data.scene_batch_sampler.scene_batch_sampler
读取配置: —
对外接口:
    - SceneBatchSampler(frame_index, batch_size, shuffle, drop_last) -> Sampler[list[int]]
说明: 跨模块统一从本入口导入。
"""

from data.scene_batch_sampler.scene_batch_sampler import SceneBatchSampler

__all__ = ["SceneBatchSampler"]
