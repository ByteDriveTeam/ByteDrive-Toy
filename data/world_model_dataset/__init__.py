"""世界模型离线栅格数据集的稳定公开入口。

模块: data/world_model_dataset/__init__.py
依赖: data.world_model_dataset.world_model_dataset
读取配置: —（实现通过 cfg 读取）
对外接口:
    - WorldModelDataset(cfg) -> torch.utils.data.Dataset
    - decode_grid(blob, shape) -> ndarray
"""

from data.world_model_dataset.world_model_dataset import WorldModelDataset, decode_grid

__all__ = ["WorldModelDataset", "decode_grid"]
