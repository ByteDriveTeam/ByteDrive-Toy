"""DINO 多层序列融合：对选定层逐层 RMSNorm 后沿末维拼接，再线性降到预测主干工作维。

模块: model/feature_fusion/feature_fusion.py
依赖: torch, model.attention.RMSNormTokens, model.feature_fusion.checks.feature_fusion_checks
读取配置: —（hidden_dim / num_layers / out_channels 由调用方以参数传入，自身不读 config）
对外接口:
    - DinoFeatureFusion(hidden_dim, num_layers, out_channels) -> nn.Module
        # forward([N,L,S,hidden]) -> [N,S,out_channels]
说明: 不同 ViT 层激活尺度差异大，先各自 RMSNorm 对齐尺度再拼接，避免深层大幅值淹没浅层纹理；
      拼接后用 Linear 把 L·hidden 融合并降到特征主干工作维（喂给 FeatureTrunk）。
      归一化统计恒用 FP32，但不改变 CLS/register/patch 的 Token 数量和顺序；
      作为可训练前端接冻结骨干之后，梯度只回传本模块与下游，不入骨干。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.attention import RMSNormTokens
from model.feature_fusion.checks.feature_fusion_checks import check_fusion_features


__all__ = ["DinoFeatureFusion"]


class DinoFeatureFusion(nn.Module):
    """逐层 RMSNorm + 拼接 + Linear 降维的 DINO 多层完整序列融合。

    Args:
        hidden_dim: 单层特征通道数（骨干 hidden_dim）。
        num_layers: 参与融合的层数 L（= len(feature_layers)）。
        out_channels: 融合输出通道（= 时序主干工作维 channels）。

    Shape:
        输入: `[N,L,S,hidden]`。
        输出: `[N,S,out_channels]`，S 及 Token 顺序不变。
    """

    def __init__(self, hidden_dim: int, num_layers: int, out_channels: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.norms = nn.ModuleList([RMSNormTokens(hidden_dim) for _ in range(num_layers)])
        self.reduce = nn.Linear(num_layers * hidden_dim, out_channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """逐层归一化对齐尺度后沿末维拼接，再线性融合降维。"""
        check_fusion_features(features, self.num_layers, self.hidden_dim)
        normed = torch.cat([norm(features[:, i]) for i, norm in enumerate(self.norms)], dim=-1)
        return self.reduce(normed)
