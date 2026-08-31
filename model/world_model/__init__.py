"""密集残差时空世界模型的稳定公开入口。

模块: model/world_model/__init__.py
依赖: model.world_model.world_model
读取配置: —（实现通过 cfg 读取）
对外接口:
    - WorldModel(cfg) -> nn.Module
    - sample_consistent_mask(batch_size, cfg, device, generator=None) -> Tensor
"""

from model.world_model.world_model import WorldModel, sample_consistent_mask

__all__ = ["WorldModel", "sample_consistent_mask"]
