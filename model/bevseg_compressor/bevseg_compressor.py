"""BEVSeg 编码、分层离散采样与 PixelShuffle 解码网络。

模块: model/bevseg_compressor/bevseg_compressor.py
依赖: torch, model.attention, model.residual_block
读取配置: model.bevseg, data.bevseg
对外接口:
    - BEVSegCompressor(cfg) -> nn.Module
      encode(x, epoch=None, sample=True) -> dict
      decode(codes) -> Tensor
      forward(x, epoch=None, sample=True) -> dict
说明: PixelShuffle 展开卷积使用 ICNR 初始化，避免随机子像素排列造成初始棋盘格。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.attention import RMSNormTokens
from model.residual_block import ResidualBlock
from model.bevseg_compressor.checks.bevseg_compressor_checks import (
    check_bevseg_input,
    check_bevseg_outputs,
)


__all__ = ["BEVSegCompressor"]


def _icnr_(weight: torch.Tensor, scale: int = 2) -> None:
    """用重复的子像素核初始化 Conv2d，使 PixelShuffle 初始近似最近邻复制。"""
    out_channels, in_channels, kh, kw = weight.shape
    if out_channels % (scale * scale) != 0:
        raise ValueError("ICNR 输出通道必须可被 PixelShuffle 倍率平方整除")
    base = torch.empty(out_channels // (scale * scale), in_channels, kh, kw,
                       device=weight.device, dtype=weight.dtype)
    nn.init.kaiming_normal_(base, mode="fan_out", nonlinearity="relu")
    with torch.no_grad():
        weight.copy_(base.repeat_interleave(scale * scale, dim=0))


class _PixelShuffleStage(nn.Module):
    """带 ICNR 初始化的 2 倍 PixelShuffle 残差阶段。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.expand = nn.Conv2d(in_channels, out_channels * 4, kernel_size=1)
        _icnr_(self.expand.weight)
        # ICNR 只要约束子像素权重复；解码扩展 bias 置零可避免初始阶段的周期性亮度偏置。
        nn.init.zeros_(self.expand.bias)
        self.shuffle = nn.PixelShuffle(2)
        self.spatial = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.mix1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.mix2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.shuffle(self.expand(x))
        out = self.spatial(out)
        return out + self.mix2(self.act(self.mix1(out)))


class BEVSegCompressor(nn.Module):
    """五帧 BEVSeg 的分层离散编码器与高保真解码器。"""

    def __init__(self, cfg) -> None:
        super().__init__()
        model_cfg = cfg.model.bevseg
        data_cfg = cfg.data.bevseg
        self.in_channels = model_cfg.in_layers * model_cfg.history_frames
        self.out_channels = self.in_channels
        self.dim = model_cfg.encoder_dim
        self.groups = model_cfg.codebook_groups
        self.vocab_size = model_cfg.codebook_size
        self.subword_dim = model_cfg.subword_dim
        self.anneal_fraction = model_cfg.anneal_fraction
        self.candidate_topk = model_cfg.candidate_topk
        self.sample_topk = model_cfg.sample_topk
        self.sharpening_start = model_cfg.sharpening_start
        self.sharpening_end = model_cfg.sharpening_end
        self.total_epochs = max(int(cfg.train.epochs), 1)

        self.patch = nn.Conv2d(self.in_channels, self.dim,
                               kernel_size=model_cfg.patch_kernel,
                               stride=model_cfg.patch_kernel)
        self.blocks16 = nn.Sequential(*(ResidualBlock(self.dim)
                                        for _ in range(model_cfg.residual_blocks_16)))
        self.down8 = nn.Conv2d(self.dim, self.dim, kernel_size=2, stride=2)
        self.blocks8 = nn.Sequential(*(ResidualBlock(self.dim)
                                       for _ in range(model_cfg.residual_blocks_8)))
        self.down4 = nn.Conv2d(self.dim, self.dim, kernel_size=2, stride=2)
        self.blocks4 = nn.Sequential(*(ResidualBlock(self.dim)
                                       for _ in range(model_cfg.residual_blocks_4)))
        self.logit_head = nn.Conv2d(self.dim, self.groups * self.vocab_size, kernel_size=1)
        self.codebook = nn.Parameter(torch.randn(self.groups, self.vocab_size,
                                                 self.subword_dim) * 0.02)
        decoder_in = self.groups * self.subword_dim
        self.code_norm = RMSNormTokens(decoder_in)
        channels = list(model_cfg.decoder_channels)
        self.decoder_stem = nn.Conv2d(decoder_in, channels[0], kernel_size=1)
        stages = []
        current = channels[0]
        for target in channels:
            stages.append(_PixelShuffleStage(current, target))
            current = target
        self.decoder_stages = nn.ModuleList(stages)
        self.output_head = nn.Conv2d(current, self.out_channels, kernel_size=3, padding=1)
        self._check_resolution(data_cfg)

    def _check_resolution(self, data_cfg):
        if data_cfg.resolution != 256 or len(self.decoder_stages) != 6:
            raise ValueError("BEVSeg 当前实现要求 256 分辨率和 6 级 PixelShuffle")

    def trainable_parameters(self):
        """返回压缩器全部可训练参数。"""
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def _schedule(self, epoch):
        """返回线性退火进度和 Top4 到 Top1 的锐化系数。"""
        progress = 1.0 if epoch is None else min(max(float(epoch), 0.0) /
                                                  (self.total_epochs * self.anneal_fraction), 1.0)
        sharpening = self.sharpening_start + progress * (
            self.sharpening_end - self.sharpening_start)
        return progress, min(max(sharpening, 0.0), 1.0)

    def _sample_codes(self, logits, epoch, sample):
        """逐子词表执行 Top8 随机 Top4 和可微的 Top1 锐化。"""
        _, sharpening = self._schedule(epoch)
        base_probabilities = torch.softmax(logits, dim=-1)
        top8_probabilities, top8_indices = base_probabilities.topk(
            self.candidate_topk, dim=-1)
        if sample and self.training:
            # 保留原始 Top1，再按 Top8 原始概率无放回随机抽取其余候选。
            top1_indices = top8_indices[..., :1]
            top1_probabilities = top8_probabilities[..., :1]
            rest_probabilities = top8_probabilities[..., 1:]
            rest_sampling_probabilities = rest_probabilities.clamp_min(1e-8)
            rest_sampling_probabilities = rest_sampling_probabilities / (
                rest_sampling_probabilities.sum(-1, keepdim=True).clamp_min(1e-8))
            rest_indices = top8_indices[..., 1:]
            draw = torch.multinomial(
                rest_sampling_probabilities.reshape(
                    -1, rest_sampling_probabilities.shape[-1]),
                self.sample_topk - 1, replacement=False,
            ).view(*rest_probabilities.shape[:-1], self.sample_topk - 1)
            sampled_indices = rest_indices.gather(-1, draw)
            sampled_probabilities = rest_probabilities.gather(-1, draw)
            indices = torch.cat((top1_indices, sampled_indices), dim=-1)
            values = torch.cat((top1_probabilities, sampled_probabilities), dim=-1)
            probabilities = values / values.sum(-1, keepdim=True).clamp_min(1e-8)

            hard = torch.zeros_like(probabilities).scatter_(
                -1, probabilities.argmax(-1, keepdim=True), 1.0)
            hard_st = hard + probabilities - probabilities.detach()
            selected = (1.0 - sharpening) * probabilities + sharpening * hard_st
        else:
            indices = logits.argmax(-1, keepdim=True)
            probabilities = torch.ones_like(indices, dtype=logits.dtype)
            selected = probabilities
        full_probs = torch.zeros_like(logits).scatter(-1, indices, selected)
        full_selected = torch.zeros_like(logits).scatter(-1, indices, selected)
        code = torch.einsum("bpgv,gvd->bpgd", full_selected, self.codebook)
        selected_indices = indices.gather(
            -1, selected.argmax(-1, keepdim=True)).squeeze(-1)
        return (code.flatten(-2), full_probs, selected_indices,
                probabilities, base_probabilities, sharpening)

    def encode(self, x: torch.Tensor, epoch=None, sample=True):
        """编码输入并返回 logits、离散索引、2048 维码字和 latent。"""
        check_bevseg_input(x, self.in_channels)
        if x.ndim == 5:
            x = x.flatten(1, 2)
        latent = self.blocks4(self.down4(self.blocks8(self.down8(self.blocks16(self.patch(x))))))
        logits = self.logit_head(latent).flatten(2).transpose(1, 2)
        logits = logits.view(x.shape[0], 16, self.groups, self.vocab_size)
        (codes, probabilities, indices, sampling_probabilities,
         base_probabilities, sharpening) = self._sample_codes(logits, epoch, sample)
        codes = self.code_norm(codes)
        return {"logits": logits, "indices": indices, "codes": codes,
                "probabilities": probabilities, "latent": latent,
                "sampling_probabilities": sampling_probabilities,
                "base_probabilities": base_probabilities,
                "sharpening": sharpening}

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """把 2048 维 Patch code 通过 PixelShuffle 恢复为 256×256 logits。"""
        b = codes.shape[0]
        x = codes.transpose(1, 2).reshape(b, self.groups * self.subword_dim, 4, 4)
        x = self.decoder_stem(x)
        for stage in self.decoder_stages:
            x = stage(x)
        return self.output_head(x)

    def forward(self, x: torch.Tensor, epoch=None, sample=True):
        """执行完整压缩与重建。"""
        encoded = self.encode(x, epoch=epoch, sample=sample)
        encoded["reconstruction_logits"] = self.decode(encoded["codes"])
        check_bevseg_outputs(encoded, self.in_channels)
        return encoded
