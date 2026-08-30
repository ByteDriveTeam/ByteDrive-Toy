"""5 帧 BEV 掩码世界模型：QKNorm+3D RoPE、密集残差 Encoder、EMA Teacher 与 Predictor。

模块: model/world_model/world_model.py
依赖: copy, torch, config.schema.Config, model.rope_3d, model.swiglu, 本模块 checks
读取配置:
    model.world_model.grid.front_m / rear_m / left_m / right_m / cell_size_m / layer_names
    model.world_model.num_frames / patch_size / mask_ratio / ema_rate
    model.world_model.encoder.dim / num_layers / num_heads / mlp_ratio / rope_theta / rope_axis_dims
    model.world_model.predictor.dim / num_layers / num_heads / mlp_ratio / teacher_layer_indices
对外接口:
    - WorldModel(cfg) -> nn.Module
    - sample_consistent_mask(batch_size, cfg, device, generator=None) -> Tensor
    - sample_mask_pair(batch_size, cfg, device, generator=None) -> tuple[Tensor,Tensor]
说明: 每个 Transformer Block 含两个独立密集混合子层；SDPA 与 FFN 各自学习一列历史权重，
      Softmax 后先融合全部历史输出，再执行 Pre-Norm 残差分支。Student 物理删除 75% 空间位置并
      沿 5 帧一致广播；Teacher 保留全 Token。Predictor 仅在降到 256 维后补 MaskToken。
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.schema import Config
from model.rope_3d import RoPE3D
from model.swiglu import SwiGLU
from model.world_model.checks.world_model_checks import check_grid_input, check_spatial_mask


__all__ = ["WorldModel", "sample_consistent_mask", "sample_mask_pair"]


def sample_consistent_mask(batch_size, cfg, device, generator=None) -> torch.Tensor:
    """在单帧空间 Token 组合中等概率取固定数量掩码，时序广播由模型内部完成。"""
    wm = cfg.model.world_model
    grid = wm.grid
    height = int(round((grid.front_m + grid.rear_m) / grid.cell_size_m)) // wm.patch_size
    width = int(round((grid.left_m + grid.right_m) / grid.cell_size_m)) // wm.patch_size
    patch_count = height * width
    masked_count = int(round(patch_count * wm.mask_ratio))
    order = torch.rand(batch_size, patch_count, device=device, generator=generator).argsort(dim=1)
    mask = torch.zeros(batch_size, patch_count, dtype=torch.bool, device=device)
    return mask.scatter_(1, order[:, :masked_count], True)


def sample_mask_pair(batch_size, cfg, device, generator=None):
    """为同一 batch 生成两个掩码不同的时序一致视图。"""
    first = sample_consistent_mask(batch_size, cfg, device, generator)
    second = sample_consistent_mask(batch_size, cfg, device, generator)
    equal = torch.all(first == second, dim=1)
    if bool(equal.any()):
        second[equal] = second[equal].roll(1, dims=1)
    return first, second


class WorldModel(nn.Module):
    """Student/EMA Teacher/六层 Predictor 的完整自监督世界模型。"""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        wm = cfg.model.world_model
        self._frames = int(wm.num_frames)
        self._channels = len(wm.grid.layer_names)
        self._height = int(round((wm.grid.front_m + wm.grid.rear_m) / wm.grid.cell_size_m))
        self._width = int(round((wm.grid.left_m + wm.grid.right_m) / wm.grid.cell_size_m))
        self._patch_height = self._height // wm.patch_size
        self._patch_width = self._width // wm.patch_size
        self._patch_count = self._patch_height * self._patch_width
        positions = _token_positions(self._frames, self._patch_height, self._patch_width)
        self.register_buffer("token_positions", positions, persistent=False)

        self.student = _Encoder(wm, self._channels, self._patch_height, self._patch_width)
        self.teacher = copy.deepcopy(self.student).requires_grad_(False)
        self.predictor = _Predictor(wm, positions)
        self._ema_rate = float(wm.ema_rate)
        self.teacher.eval()

    def forward_reconstruction(self, grids, spatial_mask):
        """执行掩码补全，返回四层预测/Teacher 目标、全时序 mask 与 Student GAP。"""
        self._check_inputs(grids, spatial_mask)
        student_final, student_taps, visible_positions = self.student(grids, spatial_mask)
        predictions = self.predictor(student_taps, spatial_mask, self._frames, self._patch_count)
        with torch.no_grad():
            _, teacher_taps, _ = self.teacher(grids, None)
        full_mask = spatial_mask[:, None].expand(-1, self._frames, -1).reshape(grids.shape[0], -1)
        return {
            "predictions": predictions,
            "targets": torch.stack(teacher_taps),
            "mask": full_mask,
            "positions": self.token_positions,
            "student_gap": student_final.mean(1),
            "visible_positions": visible_positions,
        }

    def encode_gap(self, grids, spatial_mask):
        """编码一个掩码视图并对全部可见时空 Token 做全局平均池化。"""
        self._check_inputs(grids, spatial_mask)
        final, _, _ = self.student(grids, spatial_mask)
        return final.mean(1)

    @torch.no_grad()
    def update_teacher(self) -> None:
        """以 teacher=(1-rate)·teacher+rate·student 做一次 EMA 更新。"""
        teacher_params = list(self.teacher.parameters())
        student_params = list(self.student.parameters())
        torch._foreach_mul_(teacher_params, 1.0 - self._ema_rate)
        torch._foreach_add_(teacher_params, student_params, alpha=self._ema_rate)
        for teacher_buffer, student_buffer in zip(self.teacher.buffers(), self.student.buffers()):
            teacher_buffer.copy_(student_buffer)

    @torch.no_grad()
    def reset_teacher(self) -> None:
        """用当前 Student 精确重置 Teacher，供第三阶段初始化。"""
        self.teacher.load_state_dict(self.student.state_dict())
        self.teacher.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    def trainable_parameters(self):
        """返回排除冻结 Teacher 的可训练参数迭代器。"""
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def _check_inputs(self, grids, mask):
        check_grid_input(grids, self._channels, self._frames, self._height, self._width)
        check_spatial_mask(mask, int(grids.shape[0]), self._patch_count)


class _Encoder(nn.Module):
    def __init__(self, wm, channels, patch_height, patch_width) -> None:
        super().__init__()
        enc = wm.encoder
        self.patch_embed = nn.Conv2d(channels, enc.dim, wm.patch_size, stride=wm.patch_size)
        self.transformer = _DenseTransformer(
            enc.dim, enc.num_layers, enc.num_heads, enc.mlp_ratio,
            enc.rope_axis_dims, enc.rope_theta)
        self.tap_indices = tuple(index - 1 for index in wm.predictor.teacher_layer_indices)
        self.tap_norms = nn.ModuleList([_RMSNorm(enc.dim) for _ in self.tap_indices])
        self.output_norm = _RMSNorm(enc.dim)
        self._frames = int(wm.num_frames)
        self._patch_count = patch_height * patch_width
        self.register_buffer("positions", _token_positions(self._frames, patch_height, patch_width),
                             persistent=False)

    def forward(self, grids, spatial_mask):
        batch, frames = grids.shape[:2]
        patches = self.patch_embed(grids.flatten(0, 1).float())
        tokens = patches.flatten(2).transpose(1, 2).reshape(batch, frames * self._patch_count, -1)
        positions = self.positions
        if spatial_mask is not None:
            visible = (~spatial_mask[:, None]).expand(-1, self._frames, -1).reshape(batch, -1)
            tokens = tokens[visible].reshape(batch, -1, tokens.shape[-1])
            positions = positions[None].expand(batch, -1, -1)[visible].reshape(batch, -1, 3)
        final, blocks = self.transformer(tokens, positions)
        taps = [norm(blocks[index]) for norm, index in zip(self.tap_norms, self.tap_indices)]
        return self.output_norm(final), taps, positions


class _Predictor(nn.Module):
    def __init__(self, wm, positions) -> None:
        super().__init__()
        enc, pred = wm.encoder, wm.predictor
        self.reduce = nn.Linear(len(pred.teacher_layer_indices) * enc.dim, pred.dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred.dim))
        self.transformer = _DenseTransformer(
            pred.dim, pred.num_layers, pred.num_heads, pred.mlp_ratio,
            enc.rope_axis_dims, enc.rope_theta)
        self.expand = nn.Linear(pred.dim, len(pred.teacher_layer_indices) * enc.dim)
        self.output_norms = nn.ModuleList(
            [_RMSNorm(enc.dim) for _ in pred.teacher_layer_indices])
        self.register_buffer("positions", positions, persistent=False)

    def forward(self, taps, spatial_mask, frames, patch_count):
        visible_features = self.reduce(torch.cat(taps, dim=-1))
        batch = int(spatial_mask.shape[0])
        full_mask = spatial_mask[:, None].expand(-1, frames, -1).reshape(batch, frames * patch_count)
        full = self.mask_token.expand(batch, frames * patch_count, -1).clone()
        full[~full_mask] = visible_features.reshape(-1, visible_features.shape[-1])
        predicted, _ = self.transformer(full, self.positions)
        chunks = self.expand(predicted).chunk(len(self.output_norms), dim=-1)
        return torch.stack([norm(chunk) for norm, chunk in zip(self.output_norms, chunks)])


class _DenseTransformer(nn.Module):
    def __init__(self, dim, layers, heads, mlp_ratio, axis_dims, theta) -> None:
        super().__init__()
        self.attention_norms = nn.ModuleList([_RMSNorm(dim) for _ in range(layers)])
        self.ffn_norms = nn.ModuleList([_RMSNorm(dim) for _ in range(layers)])
        self.attentions = nn.ModuleList(
            [_QKNormAttention(dim, heads, axis_dims, theta) for _ in range(layers)])
        self.ffns = nn.ModuleList([_SwiGLUFFN(dim, mlp_ratio) for _ in range(layers)])
        self.mixing_logits = nn.ParameterList([
            nn.Parameter(torch.zeros(index + 1)) for index in range(layers * 2)
        ])

    def forward(self, tokens, positions):
        history = [tokens]
        block_outputs = []
        for index, (attn_norm, ffn_norm, attention, ffn) in enumerate(zip(
                self.attention_norms, self.ffn_norms, self.attentions, self.ffns)):
            attention_input = _mix_history(history, self.mixing_logits[2 * index])
            attention_output = attention_input + attention(attn_norm(attention_input), positions)
            history.append(attention_output)
            ffn_input = _mix_history(history, self.mixing_logits[2 * index + 1])
            tokens = ffn_input + ffn(ffn_norm(ffn_input))
            history.append(tokens)
            block_outputs.append(tokens)
        return tokens, block_outputs


class _QKNormAttention(nn.Module):
    def __init__(self, dim, heads, axis_dims, theta) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.output = nn.Linear(dim, dim)
        self.rope = RoPE3D(axis_dims, theta)

    def forward(self, tokens, positions):
        batch, count, dim = tokens.shape
        qkv = self.qkv(tokens).reshape(batch, count, 3, self.heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        query = _qk_norm(query)
        key = _qk_norm(key)
        query, key = self.rope(query, key, positions)
        attended = F.scaled_dot_product_attention(
            query.to(value.dtype), key.to(value.dtype), value)
        return self.output(attended.transpose(1, 2).reshape(batch, count, dim))


class _SwiGLUFFN(nn.Module):
    def __init__(self, dim, ratio) -> None:
        super().__init__()
        expanded = dim * ratio
        self.input = nn.Linear(dim, expanded)
        self.activation = SwiGLU(dim=-1)
        self.output = nn.Linear(expanded // 2, dim)

    def forward(self, tokens):
        return self.output(self.activation(self.input(tokens)))


class _RMSNorm(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, tokens):
        dtype = tokens.dtype
        normalized = tokens.float() * torch.rsqrt(tokens.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        return (normalized * self.weight.float()).to(dtype)


def _qk_norm(features):
    dtype = features.dtype
    normalized = features.float() * torch.rsqrt(features.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    return normalized.to(dtype)


def _mix_history(history, logits):
    weights = logits.softmax(0).to(history[0].dtype)
    return torch.einsum("s,sbnd->bnd", weights, torch.stack(history))


def _token_positions(frames, height, width):
    time, row, column = torch.meshgrid(
        torch.arange(frames, dtype=torch.float32),
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32), indexing="ij")
    return torch.stack((row, column, time), dim=-1).reshape(-1, 3)
