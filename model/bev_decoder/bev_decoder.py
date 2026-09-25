"""统一 BEV 解码头：共享上采样并分别输出场景/Agent 占用与二维场。

模块: model/bev_decoder/bev_decoder.py
依赖: torch, config.schema.DrivingCfg, model.residual_block.ResidualBlock,
      model.bev_upsampler.BevUpsampler, model.bev_decoder.checks.bev_decoder_checks
读取配置:
    model.driving.work_dim
    model.driving.bev_decoder.reduce_channels / up_channels / feature_channels
    model.driving.occupancy.voxel_size_m
对外接口:
    - BevDecoder(cfg_driving) -> nn.Module
        forward(bev_feat) -> dict[str, Tensor]   # 双占用、行驶概率、车道线、停止线 logits
说明: 占用每个高度层视作独立通道；场景和 Agent 使用独立预测头。
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from config.schema import DrivingCfg
from model.bev_decoder.checks.bev_decoder_checks import check_bev_features
from model.bev_upsampler import BevUpsampler
from model.residual_block import ResidualBlock


__all__ = ["BevDecoder"]

_FIELD_NAMES = ("drivable", "lane_occupancy")


class BevDecoder(nn.Module):
    """把 BEV 骨干特征一次上采样为全部空间驾驶任务输出。"""

    def __init__(self, cfg_driving: DrivingCfg) -> None:
        super().__init__()
        self.work_dim = cfg_driving.work_dim
        decoder = cfg_driving.bev_decoder
        self.residual = ResidualBlock(self.work_dim)
        self.reduce = nn.Conv2d(
            self.work_dim, decoder.reduce_channels, kernel_size=1)
        self.upsampler = BevUpsampler(
            decoder.reduce_channels, decoder.up_channels, decoder.feature_channels)

        self.field_heads = nn.ModuleDict({
            name: nn.Conv2d(decoder.feature_channels, 1, kernel_size=1)
            for name in _FIELD_NAMES
        })
        self.stop_line_head = nn.Conv2d(
            decoder.feature_channels, 1, kernel_size=1)
        nn.init.zeros_(self.stop_line_head.weight)
        nn.init.zeros_(self.stop_line_head.bias)
        bev = cfg_driving.bev
        layers = int(round((bev.z_max_m - bev.z_min_m) / cfg_driving.occupancy.voxel_size_m))
        self.scene_occ_head = nn.Conv2d(decoder.feature_channels, layers, 1)
        self.agent_occ_head = nn.Conv2d(decoder.feature_channels, layers, 1)

    def forward(self, bev_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """返回共享空间特征解码出的全部 BEV logits 与方向向量。"""
        check_bev_features(bev_feat, self.work_dim)
        shared = self.upsampler(self.reduce(self.residual(bev_feat)))
        outputs = {name: head(shared) for name, head in self.field_heads.items()}
        outputs.update({
            "stop_line_logits": self.stop_line_head(shared),
            "scene_occ_logits": nn.functional.interpolate(
                self.scene_occ_head(shared), scale_factor=.5, mode="bilinear", align_corners=False),
            "agent_occ_logits": nn.functional.interpolate(
                self.agent_occ_head(shared), scale_factor=.5, mode="bilinear", align_corners=False),
        })
        return outputs
