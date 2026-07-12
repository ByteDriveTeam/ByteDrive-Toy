"""通用 SwiGLU 激活模块。

模块: model/swiglu/swiglu.py
依赖: torch, model.swiglu.checks.swiglu_checks
读取配置: —（激活维度由调用方以参数传入，自身不读 config）
对外接口:
    - swiglu(features, dim=-1) -> Tensor   # 沿 dim 二等分为 value/gate 后输出 value * silu(gate)
    - SwiGLU(dim=-1) -> nn.Module          # 上述激活的层封装
说明: SwiGLU 沿指定维度把通道二等分为 value 与 gate；入参校验下沉到 swiglu_checks（规范 §7.1）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.swiglu.checks.swiglu_checks import check_swiglu_dim, check_swiglu_features


__all__ = ["SwiGLU", "swiglu"]


def swiglu(features: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """沿指定维度执行 SwiGLU 激活。

    Args:
        features: 待激活特征，指定维度会被二等分为 value 和 gate。
        dim: 拆分维度。

    Returns:
        激活后的特征，指定维度长度为输入的一半。

    Shape:
        输入: `[..., 2 * C, ...]`
        输出: `[..., C, ...]`
    """

    normalized_dim = check_swiglu_features(features, dim)
    value_features, gate_features = features.chunk(2, dim=normalized_dim)
    return value_features * F.silu(gate_features)


class SwiGLU(nn.Module):
    """SwiGLU 激活层。

    Args:
        dim: 拆分 value 和 gate 的维度。

    Shape:
        输入: `[..., 2 * C, ...]`
        输出: `[..., C, ...]`
    """

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        check_swiglu_dim(dim)
        self.dim = dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """沿配置维度执行 SwiGLU 激活。"""

        return swiglu(features, dim=self.dim)
