"""Pre-Norm 交叉/自注意力块（多头，PyTorch 原生 SDPA）。

模块: model/attention/attention.py
依赖: torch, model.swiglu.SwiGLU, model.attention.checks.attention_checks
读取配置: —（dim / num_heads / mlp_ratio 由调用方以参数传入，来源为 config.model.driving.attention）
对外接口:
    - CrossAttentionBlock(dim, num_heads, mlp_ratio) -> nn.Module   # forward(query, context) -> query'
    - SelfAttentionBlock(dim, num_heads, mlp_ratio) -> nn.Module    # forward(x) -> x'
说明: Token 序列为通道末置 `[B, N, C]`。Pre-Norm 结构（RMSNorm 在残差分支前），注意力用
      `F.scaled_dot_product_attention`（SDPA，内部走 FlashAttention/内存高效核）。前馈用 SwiGLU
      （复用 model.swiglu，不重复造轮子）。交叉注意力对 query 与 context 各自归一；自注意力
      query=context。RMSNorm 均方根统计恒 FP32 再回落输入精度（与 residual_block 一致）；本模块不
      强制整体精度，混精边界由外层 autocast 控制。供 BEV↔图像、Token↔BEV 复用。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention.checks.attention_checks import check_attention_args, check_token_features
from model.swiglu import SwiGLU


__all__ = ["CrossAttentionBlock", "SelfAttentionBlock"]


class _RMSNormLast(nn.Module):
    """对 Token 序列末维（通道）做 RMSNorm；统计量恒 FP32 再回落输入精度。"""

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

    def forward(self, q_in: torch.Tensor, kv_in: torch.Tensor) -> torch.Tensor:
        b, nq, c = q_in.shape
        q = self._split(self.q_proj(q_in))
        k = self._split(self.k_proj(kv_in))
        v = self._split(self.v_proj(kv_in))
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
        self.norm_q = _RMSNormLast(dim)
        self.norm_kv = _RMSNormLast(dim)
        self.attn = _MultiHeadAttention(dim, num_heads)
        self.norm_ffn = _RMSNormLast(dim)
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
        self.norm = _RMSNormLast(dim)
        self.attn = _MultiHeadAttention(dim, num_heads)
        self.norm_ffn = _RMSNormLast(dim)
        self.ffn = _FeedForward(dim, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-Norm 自注意力残差 + Pre-Norm 前馈残差。"""
        check_token_features(x, self.dim, "x")
        h = self.norm(x)
        x = x + self.attn(h, h)
        x = x + self.ffn(self.norm_ffn(x))
        return x
