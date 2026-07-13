"""Pre-Norm 交叉/自注意力块（多头，PyTorch 原生 SDPA）：公开 API 重导出入口。

模块: model/attention/__init__.py
依赖: model.attention.attention
读取配置: —
对外接口:
    - CrossAttentionBlock(dim, num_heads, mlp_ratio) -> nn.Module   # forward(query, context)
    - SelfAttentionBlock(dim, num_heads, mlp_ratio) -> nn.Module    # forward(x)
说明: 跨模块统一 `from model.attention import ...`；实现见 attention.py，入参校验见 checks/。
"""

from model.attention.attention import CrossAttentionBlock, SelfAttentionBlock

__all__ = ["CrossAttentionBlock", "SelfAttentionBlock"]
