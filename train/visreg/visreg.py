"""VISReg：对多视图 GAP 特征施加不变性、中心、尺度和切片 Wasserstein 形状约束。

模块: train/visreg/visreg.py
依赖: math, torch, config.schema.Config, 本模块 checks
读取配置:
    train.world_model.visreg.tradeoff / slices / center_weight / scale_weight / shape_weight
对外接口:
    - VISRegLoss(cfg) -> nn.Module
说明: 严格按 VISReg 官方实现使用总体标准差（除 sqrt(B)）、stop-gradient 尺度归一与标准
      高斯固定分位数；形状项和目标恒 FP32，避免 AMP 下分位数精度损失。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.schema import Config
from train.visreg.checks.visreg_checks import check_visreg_features


__all__ = ["VISRegLoss"]


class VISRegLoss(nn.Module):
    """计算两视图不变性与 VISReg 三项分布正则。"""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        visreg = cfg.train.world_model.visreg
        self.tradeoff = float(visreg.tradeoff)
        self.slices = int(visreg.slices)
        self.center_weight = float(visreg.center_weight)
        self.scale_weight = float(visreg.scale_weight)
        self.shape_weight = float(visreg.shape_weight)

    def forward(self, features):
        """返回总损失及可记录的各分量；features 为 `[V,B,D]` GAP。"""
        check_visreg_features(features)
        values = features.float()
        mean = values.mean(dim=1, keepdim=True)
        centered = values - mean
        std = centered.norm(dim=1).div(math.sqrt(values.shape[1])).clamp_min(1e-6)
        normalized = centered / std.detach().unsqueeze(1)
        directions = F.normalize(
            torch.randn(values.shape[-1], self.slices, device=values.device, dtype=torch.float32),
            dim=0)
        projected = (normalized @ directions).sort(dim=1).values
        quantiles = torch.linspace(
            1, values.shape[1], values.shape[1], device=values.device, dtype=torch.float32)
        target = torch.erfinv(2.0 * quantiles / (values.shape[1] + 1) - 1.0) * math.sqrt(2.0)
        components = {
            "visreg_invariance": (values[0] - values[1]).pow(2).mean(),
            "visreg_center": mean.pow(2).mean(),
            "visreg_scale": (std - 1.0).pow(2).mean(),
            "visreg_shape": (projected - target.view(1, -1, 1)).pow(2).mean(),
        }
        regularization = (self.center_weight * components["visreg_center"]
                          + self.scale_weight * components["visreg_scale"]
                          + self.shape_weight * components["visreg_shape"])
        total = ((1.0 - self.tradeoff) * components["visreg_invariance"]
                 + self.tradeoff * regularization)
        return total, components
