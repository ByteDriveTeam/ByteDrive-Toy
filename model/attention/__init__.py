"""Pre-Norm 交叉/自注意力块（多头，含 patch-only 2D RoPE）：公开 API 重导出入口。

模块: model/attention/__init__.py
依赖: model.attention.attention
读取配置: —
对外接口:
    - RMSNormTokens(dim) -> nn.Module   # Token 末维 RMSNorm
    - CrossAttentionBlock(dim, num_heads, mlp_ratio) -> nn.Module   # forward(query, context)
    - SelfAttentionBlock(dim, num_heads, mlp_ratio) -> nn.Module    # forward(x)
    - ImageSelfAttentionBlock(dim, num_heads, mlp_ratio, rope_theta) -> nn.Module
        # forward(x, patch_positions)
说明: 跨模块统一 `from model.attention import ...`；实现见 attention.py，入参校验见 checks/。
"""

from model.attention.attention import (
    CrossAttentionBlock,
    ImageSelfAttentionBlock,
    RMSNormTokens,
    SelfAttentionBlock,
)

__all__ = ["RMSNormTokens", "CrossAttentionBlock", "SelfAttentionBlock", "ImageSelfAttentionBlock"]
