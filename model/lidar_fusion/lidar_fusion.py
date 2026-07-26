"""LiDAR 体素融合：lg-Symlog 编码三维统计，并以视觉条件门控注入初始 BEV 查询。

模块: model/lidar_fusion/lidar_fusion.py
依赖: contextlib, torch, config.schema.DrivingCfg,
      model.lidar_fusion.checks.lidar_fusion_checks
读取配置:
    model.driving.work_dim
    model.driving.bev.x_min_m / x_max_m / y_min_m / y_max_m / z_min_m / z_max_m / height / width
    model.driving.lidar_fusion.voxel_size_m / voxel_embed_dim / height_hidden_dim /
        reduced_dim / gate_hidden_dim
对外接口:
    - LidarQueryFusion(cfg_driving) -> nn.Module
        forward(query, visual, stats=None, occupied=None, valid=None) -> Tensor
说明: 空体素由可学习向量表示；整帧缺失由 valid 严格旁路。门控为逐位置逐通道 Sigmoid，
      最终空间对齐卷积零初始化，使旧权重初始化时 LiDAR 残差严格为零。
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn

from config.schema import DrivingCfg
from model.lidar_fusion.checks.lidar_fusion_checks import check_lidar_fusion_inputs


__all__ = ["LidarQueryFusion"]


class LidarQueryFusion(nn.Module):
    """把 `[B,6,Z,X,Y]` LiDAR 统计编码并门控注入初始 BEV 查询。"""

    def __init__(self, cfg_driving: DrivingCfg) -> None:
        super().__init__()
        bev, cfg = cfg_driving.bev, cfg_driving.lidar_fusion
        self.work_dim = cfg_driving.work_dim
        self.grid_shape = (
            int(round((bev.z_max_m - bev.z_min_m) / cfg.voxel_size_m)),
            int(round((bev.x_max_m - bev.x_min_m) / cfg.voxel_size_m)),
            int(round((bev.y_max_m - bev.y_min_m) / cfg.voxel_size_m)),
        )
        self.bev_shape = (bev.height, bev.width)
        z_count, x_count, y_count = self.grid_shape
        align_factor = x_count // bev.height
        self.empty_embedding = nn.Parameter(torch.zeros(cfg.voxel_embed_dim))
        self.voxel_projection = nn.Conv3d(6, cfg.voxel_embed_dim, 1)
        self.height_reducer = nn.Sequential(
            nn.Conv2d(cfg.voxel_embed_dim * z_count, cfg.height_hidden_dim, 1),
            nn.SiLU(),
            nn.Conv2d(cfg.height_hidden_dim, cfg.reduced_dim, 1),
        )
        self.spatial_alignment = nn.Conv2d(
            cfg.reduced_dim, self.work_dim, align_factor, stride=align_factor)
        self.gate = nn.Sequential(
            nn.Linear(self.work_dim * 2, cfg.gate_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.gate_hidden_dim, self.work_dim),
        )
        nn.init.zeros_(self.spatial_alignment.weight)
        nn.init.zeros_(self.spatial_alignment.bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, query, visual, stats=None, occupied=None, valid=None):
        """返回 LiDAR 残差融合后的 BEV 查询；未提供或全无效时严格旁路。"""
        if stats is None or occupied is None or valid is None:
            return query
        check_lidar_fusion_inputs(
            query, visual, stats, occupied, valid, self.work_dim,
            self.grid_shape, self.bev_shape)
        if not bool(valid.any()):
            return query

        with _fp32_context(stats.device):
            stats_fp32 = stats.float()
            symlog = torch.sign(stats_fp32) * torch.log10(1.0 + stats_fp32.abs())
        encoded = self.voxel_projection(symlog)
        empty = self.empty_embedding.to(encoded.dtype)[None, :, None, None, None]
        encoded = torch.where(occupied, encoded, empty)
        batch, channels, depth, height, width = encoded.shape
        planar = encoded.reshape(batch, channels * depth, height, width)
        lidar_feature = self.spatial_alignment(self.height_reducer(planar))

        visual_global = visual.mean(dim=(1, 3, 4))
        local = lidar_feature.permute(0, 2, 3, 1)
        global_grid = visual_global[:, None, None, :].expand(
            -1, local.shape[1], local.shape[2], -1)
        gate = torch.sigmoid(self.gate(torch.cat((global_grid, local), dim=-1)))
        weighted = (gate * local).permute(0, 3, 1, 2)
        frame_valid = valid.to(weighted.dtype)[:, None, None, None]
        return query + frame_valid * weighted


def _fp32_context(device):
    """仅在数值变换段关闭外层 autocast，其他设备回退为空上下文。"""
    if device.type == "meta":
        return nullcontext()
    try:
        return torch.autocast(device_type=device.type, enabled=False)
    except (RuntimeError, ValueError):
        return nullcontext()
