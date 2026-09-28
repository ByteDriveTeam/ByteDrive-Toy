"""特权 BEV 的连续 latent 变分自编码器。

模块: model/privileged_bev_vae/privileged_bev_vae.py
依赖: torch, model.residual_block, model.bevseg_compressor
读取配置: model.privileged_bev_vae.*, data.bevseg.resolution
对外接口:
    - PrivilegedBEVVAE(cfg) -> nn.Module
      encode(x, sample=None) -> dict
      decode(z) -> Tensor
      forward(x, sample=None) -> dict
说明: 编码器四次 stride-2 下采样得到 16 倍空间压缩；连续 latent 由 RMSNorm2d
      后的均值和对数方差投影产生，解码器不在首次上采样前额外归一化。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.bevseg_compressor.bevseg_compressor import _PixelShuffleStage
from model.privileged_bev_vae.checks.privileged_bev_vae_checks import (
    check_vae_input,
    check_vae_outputs,
)
from model.residual_block import RMSNorm2d, ResidualBlock


__all__ = ["PrivilegedBEVVAE"]


class PrivilegedBEVVAE(nn.Module):
    """将 BEVSeg 特权栅格编码为连续高斯 latent 并重建栅格。"""

    def __init__(self, cfg) -> None:
        super().__init__()
        model_cfg = cfg.model.privileged_bev_vae
        data_cfg = cfg.data.bevseg
        self.in_channels = int(model_cfg.in_layers)
        self.encoder_dim = int(model_cfg.encoder_dim)
        self.latent_dim = int(model_cfg.latent_dim)
        self.resolution = int(data_cfg.resolution)
        self.downsample_kernel = int(model_cfg.downsample_kernel)
        self.downsample_stages = len(model_cfg.encoder_residual_blocks)
        self.latent_resolution = self.resolution // (2 ** self.downsample_stages)

        self.encoder = nn.ModuleList()
        channels = self.in_channels
        for residual_count in model_cfg.encoder_residual_blocks:
            self.encoder.append(nn.Conv2d(
                channels, self.encoder_dim, kernel_size=self.downsample_kernel,
                stride=2))
            self.encoder.append(nn.Sequential(*(
                ResidualBlock(self.encoder_dim) for _ in range(residual_count))))
            channels = self.encoder_dim
        self.latent_norm = RMSNorm2d(self.encoder_dim)
        self.mu_head = nn.Conv2d(self.encoder_dim, self.latent_dim, kernel_size=1)
        self.logvar_head = nn.Conv2d(self.encoder_dim, self.latent_dim, kernel_size=1)

        decoder_channels = list(model_cfg.decoder_channels)
        self.decoder_stem = nn.Conv2d(self.latent_dim, decoder_channels[0], kernel_size=1)
        stages = []
        current = decoder_channels[0]
        for target in decoder_channels:
            stages.append(_PixelShuffleStage(current, target))
            current = target
        self.decoder_stages = nn.ModuleList(stages)
        self.output_head = nn.Conv2d(current, self.in_channels, kernel_size=3, padding=1)

    def trainable_parameters(self):
        """返回 VAE 的全部可训练参数。"""
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def encode(self, x: torch.Tensor, sample=None):
        """编码输入并返回均值、对数方差与连续 latent。"""
        check_vae_input(x, self.in_channels, self.resolution)
        features = x
        for layer in self.encoder:
            features = layer(features)
        normalized = self.latent_norm(features)
        mu = self.mu_head(normalized)
        logvar = self.logvar_head(normalized).clamp(-30.0, 20.0)
        if sample is None:
            sample = self.training
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else mu
        return {"mu": mu, "logvar": logvar, "z": z, "features": features}

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """从连续 latent 重建 BEVSeg logits。"""
        expected = (self.latent_dim, self.latent_resolution, self.latent_resolution)
        if z.ndim != 4 or tuple(z.shape[1:]) != expected:
            raise ValueError("特权 BEV VAE latent 空间形状与配置不一致")
        x = self.decoder_stem(z)
        for stage in self.decoder_stages:
            x = stage(x)
        return self.output_head(x)

    def forward(self, x: torch.Tensor, sample=None):
        """执行编码、重参数采样和重建。"""
        encoded = self.encode(x, sample=sample)
        encoded["reconstruction_logits"] = self.decode(encoded["z"])
        check_vae_outputs(
            encoded, self.in_channels, self.latent_dim,
            self.latent_resolution, self.resolution)
        return encoded
