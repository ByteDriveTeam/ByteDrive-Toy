"""整幅 BEV 64 倍压缩 VAE 的训练循环适配器。

模块: train/bevseg_vae64/bevseg_vae64.py
依赖: train.privileged_bev_vae
读取配置: train.bevseg_vae64.*
对外接口:
    - train_bevseg_vae64_epoch(model, loader, optimizer, cfg, device, epoch) -> dict
说明: 训练损失和梯度流程与现有特权 BEV VAE 一致，仅切换到新增配置项。
"""

from train.privileged_bev_vae.privileged_bev_vae import (
    train_privileged_bev_vae_epoch,
)

__all__ = ["train_bevseg_vae64_epoch"]


def train_bevseg_vae64_epoch(model, loader, optimizer, cfg, device, epoch=0):
    """训练一个整幅 BEV VAE epoch，并返回样本加权平均损失。"""
    return train_privileged_bev_vae_epoch(
        model, loader, optimizer, cfg, device, epoch=epoch,
        train_config_key="bevseg_vae64", log_prefix="bevseg_vae64")
