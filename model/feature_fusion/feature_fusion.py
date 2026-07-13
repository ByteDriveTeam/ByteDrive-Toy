"""DINO 多层特征融合：对选定层逐层 RMSNorm 后沿通道拼接，再 1×1 卷积降到时序主干工作维。

模块: model/feature_fusion/feature_fusion.py
依赖: torch, model.residual_block.RMSNorm2d, model.feature_fusion.checks.feature_fusion_checks
读取配置: —（hidden_dim / num_layers / out_channels 由调用方以参数传入，自身不读 config）
对外接口:
    - DinoFeatureFusion(hidden_dim, num_layers, out_channels) -> nn.Module
        # forward([N,L,hidden,gh,gw]) -> [N,out_channels,gh,gw]
说明: 不同 ViT 层激活尺度差异大，先各自 RMSNorm 对齐尺度再拼接，避免深层大幅值淹没浅层纹理；
      拼接后用 1×1 逐点卷积把 L·hidden 融合并降到特征主干工作维（喂给 FeatureTrunk）。
      RMSNorm 复用 residual_block 的 RMSNorm2d（均方根统计恒 FP32），本模块不重复造轮子；
      作为可训练前端接冻结骨干之后，梯度只回传本模块与下游，不入骨干。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.feature_fusion.checks.feature_fusion_checks import check_fusion_features
from model.residual_block import RMSNorm2d


__all__ = ["DinoFeatureFusion"]


class DinoFeatureFusion(nn.Module):
    """逐层 RMSNorm + 拼接 + 1×1 卷积降维的 DINO 多层特征融合。

    Args:
        hidden_dim: 单层特征通道数（骨干 hidden_dim）。
        num_layers: 参与融合的层数 L（= len(feature_layers)）。
        out_channels: 融合输出通道（= 时序主干工作维 channels）。

    Shape:
        输入: `[N, L, hidden, gh, gw]`。
        输出: `[N, out_channels, gh, gw]`。
    """

    def __init__(self, hidden_dim: int, num_layers: int, out_channels: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # 每层独立 RMSNorm：各层激活尺度不同，独立可学习缩放对齐后再拼接
        self.norms = nn.ModuleList([RMSNorm2d(hidden_dim) for _ in range(num_layers)])
        # 拼接后 1×1 逐点卷积把 L·hidden 融合降到工作维
        self.reduce = nn.Conv2d(num_layers * hidden_dim, out_channels, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """逐层归一化对齐尺度后沿通道拼接，再 1×1 卷积融合降维。"""
        check_fusion_features(features, self.num_layers, self.hidden_dim)
        # features[:, i] 形如 [N, hidden, gh, gw]；逐层 RMSNorm 后沿通道维拼成 [N, L·hidden, gh, gw]
        normed = torch.cat([norm(features[:, i]) for i, norm in enumerate(self.norms)], dim=1)
        return self.reduce(normed)
