"""特权 BEV VAE 的形状和配置相关运行期校验。

模块: model/privileged_bev_vae/checks/privileged_bev_vae_checks.py
依赖: torch
读取配置: 无
对外接口:
    - check_vae_input(x, in_channels, resolution) -> None
    - check_vae_outputs(outputs, in_channels, latent_dim, latent_resolution) -> None
"""

import torch


def check_vae_input(x, in_channels, resolution):
    """校验对象: PrivilegedBEVVAE.forward 的输入张量。"""
    if not isinstance(x, torch.Tensor) or x.ndim != 4:
        raise ValueError("特权 BEV VAE 输入必须为 [B,C,H,W] 四维张量")
    if x.shape[1] != in_channels or x.shape[-2:] != (resolution, resolution):
        raise ValueError("特权 BEV VAE 输入通道或空间尺寸与配置不一致")


def check_vae_outputs(outputs, in_channels, latent_dim, latent_resolution, resolution):
    """校验对象: PrivilegedBEVVAE.forward 的输出字典。"""
    expected = (latent_dim, latent_resolution, latent_resolution)
    for name in ("mu", "logvar", "z"):
        if tuple(outputs[name].shape[1:]) != expected:
            raise ValueError("特权 BEV VAE {} 形状错误".format(name))
    if tuple(outputs["reconstruction_logits"].shape[1:]) != (
            in_channels, resolution, resolution):
        raise ValueError("特权 BEV VAE 重建形状错误")
