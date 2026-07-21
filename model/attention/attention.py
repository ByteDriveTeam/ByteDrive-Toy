"""Pre-Norm 交叉/自注意力块（多头，PyTorch 原生 SDPA，可选 patch-only 2D RoPE）。

模块: model/attention/attention.py
依赖: torch, model.swiglu.SwiGLU, model.attention.checks.attention_checks
读取配置: —（dim / num_heads / mlp_ratio 由调用方以参数传入，来源为 config.model.driving.attention）
对外接口:
    - RMSNormTokens(dim) -> nn.Module   # 对 [B,N,C] 末维做 FP32 统计的 RMSNorm
    - CrossAttentionBlock(dim, num_heads, mlp_ratio) -> nn.Module   # forward(query, context) -> query'
    - SelfAttentionBlock(dim, num_heads, mlp_ratio) -> nn.Module    # forward(x) -> x'
    - ImageSelfAttentionBlock(dim, num_heads, mlp_ratio, rope_theta) -> nn.Module
        # forward(x, patch_positions) -> x'
说明: Token 序列为通道末置 `[B, N, C]`。Pre-Norm 结构（RMSNorm 在残差分支前），注意力用
      `F.scaled_dot_product_attention`（SDPA，内部走 FlashAttention/内存高效核）。前馈用 SwiGLU
      （复用 model.swiglu，不重复造轮子）。交叉注意力对 query 与 context 各自归一；自注意力
      query=context。RMSNorm 均方根统计恒 FP32 再回落输入精度（与 residual_block 一致）；本模块不
      强制整体精度，混精边界由外层 autocast 控制。ImageSelfAttentionBlock 沿用 DINOv3
      的二维 RoPE 排布，仅旋转序列尾部的 patch query/key，前缀 CLS/寄存器 Token 保持原样。
      供 BEV↔图像、Token↔BEV 及图像序列复用。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention.checks.attention_checks import (
    check_attention_args,
    check_image_attention_args,
    check_patch_positions,
    check_token_features,
)
from model.swiglu import SwiGLU


__all__ = ["RMSNormTokens", "CrossAttentionBlock", "SelfAttentionBlock", "ImageSelfAttentionBlock"]


class RMSNormTokens(nn.Module):
    """对 Token 序列末维（通道）做 RMSNorm；统计量恒 FP32 再回落输入精度。

    Args:
        dim: Token 通道维数。

    Shape:
        输入/输出: `[B,N,dim]`。
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        # eps 必须放在平方根内；sqrt(mean(x^2)) 在全零输入处的反向导数奇异，
        # 即使前向结果有限也会产生 NaN 梯度。
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * x).to(orig_dtype)


class _FeedForward(nn.Module):
    """SwiGLU 前馈：Linear 升到 expanded=mlp_ratio·C → SwiGLU 折半到 expanded/2 → Linear 回 C。

    即 C → 4C → SwiGLU → 2C → C（mlp_ratio=4）：SwiGLU 把 4C 二等分为 value/gate 得 2C 有效隐藏维，
    再降回 C。相比「先 ×2 再 SwiGLU」省一半首层参数、隐藏维更合规范（LLaMA 风格）。
    """

    def __init__(self, dim: int, mlp_ratio: int) -> None:
        super().__init__()
        expanded = dim * mlp_ratio             # 4C
        self.fc1 = nn.Linear(dim, expanded)    # C -> 4C（SwiGLU 沿末维二等分 value/gate）
        self.act = SwiGLU(dim=-1)              # 4C -> 2C
        self.fc2 = nn.Linear(expanded // 2, dim)  # 2C -> C

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _MultiHeadAttention(nn.Module):
    """多头注意力核（SDPA）：query 来自 q_in，key/value 来自 kv_in。"""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        # [B, N, C] -> [B, H, N, dh]
        b, n, _ = x.shape
        return x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        q_in: torch.Tensor,
        kv_in: torch.Tensor,
        patch_positions: torch.Tensor | None = None,
        rope_theta: float | None = None,
    ) -> torch.Tensor:
        b, nq, c = q_in.shape
        q = self._split(self.q_proj(q_in))
        k = self._split(self.k_proj(kv_in))
        v = self._split(self.v_proj(kv_in))
        if patch_positions is not None:
            q, k = _apply_patch_rope_2d(q, k, patch_positions, float(rope_theta))
        out = F.scaled_dot_product_attention(q, k, v)  # [B, H, Nq, dh]
        out = out.transpose(1, 2).reshape(b, nq, c)
        return self.o_proj(out)


class CrossAttentionBlock(nn.Module):
    """Pre-Norm 多头交叉注意力块 + SwiGLU 前馈。

    Args:
        dim: Token 通道维 C（须被 num_heads 整除）。
        num_heads: 注意力头数。
        mlp_ratio: 前馈隐藏维膨胀率。

    Shape:
        query: `[B, Nq, C]`，context: `[B, Nk, C]`，输出: `[B, Nq, C]`。
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int) -> None:
        super().__init__()
        check_attention_args(dim, num_heads, mlp_ratio)
        self.dim = dim
        self.norm_q = RMSNormTokens(dim)
        self.norm_kv = RMSNormTokens(dim)
        self.attn = _MultiHeadAttention(dim, num_heads)
        self.norm_ffn = RMSNormTokens(dim)
        self.ffn = _FeedForward(dim, mlp_ratio)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """query 查询 context：Pre-Norm 交叉注意力残差 + Pre-Norm 前馈残差。"""
        check_token_features(query, self.dim, "query")
        check_token_features(context, self.dim, "context")
        query = query + self.attn(self.norm_q(query), self.norm_kv(context))
        query = query + self.ffn(self.norm_ffn(query))
        return query


class SelfAttentionBlock(nn.Module):
    """Pre-Norm 多头自注意力块 + SwiGLU 前馈。

    Args:
        dim: Token 通道维 C（须被 num_heads 整除）。
        num_heads: 注意力头数。
        mlp_ratio: 前馈隐藏维膨胀率。

    Shape:
        输入/输出: `[B, N, C]`。
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int) -> None:
        super().__init__()
        check_attention_args(dim, num_heads, mlp_ratio)
        self.dim = dim
        self.norm = RMSNormTokens(dim)
        self.attn = _MultiHeadAttention(dim, num_heads)
        self.norm_ffn = RMSNormTokens(dim)
        self.ffn = _FeedForward(dim, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-Norm 自注意力残差 + Pre-Norm 前馈残差。"""
        check_token_features(x, self.dim, "x")
        h = self.norm(x)
        x = x + self.attn(h, h)
        x = x + self.ffn(self.norm_ffn(x))
        return x


class ImageSelfAttentionBlock(SelfAttentionBlock):
    """Pre-Norm 图像自注意力块：仅对序列尾部 patch Token 应用 DINOv3 式二维 RoPE。

    Args:
        dim: Token 通道维 C。
        num_heads: 注意力头数，每头维度须被 4 整除。
        mlp_ratio: SwiGLU 前馈膨胀率。
        rope_theta: 二维 RoPE 基频。

    Shape:
        x: `[B,N,C]`；patch_positions: `[P,2]`；输出: `[B,N,C]`。
        前 N-P 个 Token 是 CLS/寄存器等前缀，不施加位置旋转。
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int, rope_theta: float) -> None:
        super().__init__(dim, num_heads, mlp_ratio)
        check_image_attention_args(dim, num_heads, rope_theta)
        self.rope_theta = float(rope_theta)

    def forward(self, x: torch.Tensor, patch_positions: torch.Tensor) -> torch.Tensor:
        """对完整图像 Token 序列作 Pre-Norm Transformer，仅 patch query/key 编码位置。"""
        check_token_features(x, self.dim, "x")
        check_patch_positions(patch_positions, int(x.shape[1]))
        h = self.norm(x)
        x = x + self.attn(h, h, patch_positions, self.rope_theta)
        return x + self.ffn(self.norm_ffn(x))


def _apply_patch_rope_2d(
    query: torch.Tensor,
    key: torch.Tensor,
    patch_positions: torch.Tensor,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按 DINOv3 的二维频率排布旋转 patch，并原样保留序列前缀。"""
    patch_count = int(patch_positions.shape[0])
    prefix_count = int(query.shape[-2]) - patch_count
    head_dim = int(query.shape[-1])
    dtype = query.dtype

    frequency_indices = torch.arange(0, head_dim, 4, device=query.device, dtype=torch.float32)
    inv_frequencies = torch.pow(
        torch.tensor(theta, device=query.device, dtype=torch.float32),
        -frequency_indices / float(head_dim),
    )
    angles = patch_positions.float()[..., None] * inv_frequencies
    angles = angles.flatten(1).tile(1, 2)
    cos_angles = torch.cos(angles)[None, None]
    sin_angles = torch.sin(angles)[None, None]

    query_prefix, query_patches = query.float().split((prefix_count, patch_count), dim=-2)
    key_prefix, key_patches = key.float().split((prefix_count, patch_count), dim=-2)
    query_patches = query_patches * cos_angles + _rotate_half(query_patches) * sin_angles
    key_patches = key_patches * cos_angles + _rotate_half(key_patches) * sin_angles
    return (
        torch.cat((query_prefix, query_patches), dim=-2).to(dtype),
        torch.cat((key_prefix, key_patches), dim=-2).to(dtype),
    )


def _rotate_half(features: torch.Tensor) -> torch.Tensor:
    """交换前后半通道并对原后半取负，与 DINOv3 RoPE 旋转规则一致。"""
    first, second = features.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)
