"""世界模型三阶段训练循环的稳定公开入口。

模块: train/world_model_loop/__init__.py
依赖: train.world_model_loop.world_model_loop
读取配置: —（实现通过 cfg 读取）
对外接口:
    - train_world_model_epoch(model, loader, optimizer, cfg, device, stage, global_step, epoch_seed)
        -> tuple[dict,int]
"""

from train.world_model_loop.world_model_loop import train_world_model_epoch

__all__ = ["train_world_model_epoch"]
