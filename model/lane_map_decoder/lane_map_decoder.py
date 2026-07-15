"""独立道路线图解码器：BEV 特征上采样为道路线类别 logits 与有向切向量。

模块: model/lane_map_decoder/lane_map_decoder.py
依赖: torch, config.schema.DrivingCfg, model.residual_block.ResidualBlock,
      model.pixel_shuffle_upsampler.PixelShuffleUpsampler, model.lane_map_decoder.checks.lane_map_decoder_checks
读取配置:
    model.driving.work_dim
    model.driving.lane_map.class_names / reduce_channels / up_channels / feature_channels
对外接口:
    - LaneMapDecoder(cfg_driving) -> nn.Module
        forward(bev_feat) -> dict[str, Tensor]   # lane_class_logits [B,K,H,W] / lane_direction [B,2,H,W]
说明: 道路线图与风险/可行驶/分布三场使用完全独立的残差、压缩和像素洗牌上采样参数，避免细线类别与稠密场
      互相挤占解码容量。类别头输出未归一化 logits；方向头输出有符号二维向量，训练损失在道路线像素上归一化
      后监督实际行驶方向。卷积沿用 PyTorch 随机权重初始化，不使用全零初始化。
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from config.schema import DrivingCfg
from model.lane_map_decoder.checks.lane_map_decoder_checks import check_lane_map_features
from model.pixel_shuffle_upsampler import PixelShuffleUpsampler
from model.residual_block import ResidualBlock


__all__ = ["LaneMapDecoder"]


class LaneMapDecoder(nn.Module):
    """把 BEV 骨干特征独立解码为道路线类别与方向图。"""

    def __init__(self, cfg_driving: DrivingCfg) -> None:
        super().__init__()
        self.work_dim = cfg_driving.work_dim
        lane = cfg_driving.lane_map
        self.num_classes = len(lane.class_names)
        self.residual = ResidualBlock(self.work_dim)
        self.reduce = nn.Conv2d(self.work_dim, lane.reduce_channels, kernel_size=1)
        self.upsampler = PixelShuffleUpsampler(
            lane.reduce_channels, lane.up_channels, lane.feature_channels)
        self.act = nn.GELU()
        self.class_head = nn.Conv2d(lane.feature_channels, self.num_classes, kernel_size=1)
        self.direction_head = nn.Conv2d(lane.feature_channels, 2, kernel_size=1)

    def forward(self, bev_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """返回类别 logits `[B,K,H,W]` 与方向向量 `[B,2,H,W]`。"""
        check_lane_map_features(bev_feat, self.work_dim)
        shared = self.act(self.upsampler(self.reduce(self.residual(bev_feat))))
        return {
            "lane_class_logits": self.class_head(shared),
            "lane_direction": self.direction_head(shared),
        }
