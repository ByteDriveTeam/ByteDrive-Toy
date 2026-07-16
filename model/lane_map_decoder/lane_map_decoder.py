"""道路细线解码器：共享高分辨率特征输出道路线、相关停止线与交通灯状态。

模块: model/lane_map_decoder/lane_map_decoder.py
依赖: torch, config.schema.DrivingCfg, model.residual_block.ResidualBlock,
      model.pixel_shuffle_upsampler.PixelShuffleUpsampler, model.lane_map_decoder.checks.lane_map_decoder_checks
读取配置:
    model.driving.work_dim / model.driving.traffic_control.state_names
    model.driving.lane_map.class_names / reduce_channels / up_channels / feature_channels
对外接口:
    - LaneMapDecoder(cfg_driving) -> nn.Module
        forward(bev_feat) -> dict[str, Tensor]   # 道路线 + stop_line_logits + traffic_light_state_logits
说明: 道路线图与风险/可行驶/分布三场使用完全独立的残差、压缩和像素洗牌上采样参数，避免细线类别与稠密场
      互相挤占解码容量。停止线与灯色只新增末端 1×1 头，旧检查点可完整复用共享上采样部分；新增头零初始化，
      首次恢复时分别从中性二分类概率与均匀灯色概率开始，不扰动已有道路线输出。
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
    """把 BEV 骨干特征解码为道路线、相关停止线与交通灯状态。"""

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
        state_count = len(cfg_driving.traffic_control.state_names)
        self.stop_line_head = nn.Conv2d(lane.feature_channels, 1, kernel_size=1)
        self.traffic_light_state_head = nn.Conv2d(
            lane.feature_channels, state_count, kernel_size=1)
        nn.init.zeros_(self.stop_line_head.weight)
        nn.init.zeros_(self.stop_line_head.bias)
        nn.init.zeros_(self.traffic_light_state_head.weight)
        nn.init.zeros_(self.traffic_light_state_head.bias)

    def forward(self, bev_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """返回道路线/停止线/灯色 logits 与道路线方向向量。"""
        check_lane_map_features(bev_feat, self.work_dim)
        shared = self.act(self.upsampler(self.reduce(self.residual(bev_feat))))
        return {
            "lane_class_logits": self.class_head(shared),
            "lane_direction": self.direction_head(shared),
            "stop_line_logits": self.stop_line_head(shared),
            "traffic_light_state_logits": self.traffic_light_state_head(shared),
        }
