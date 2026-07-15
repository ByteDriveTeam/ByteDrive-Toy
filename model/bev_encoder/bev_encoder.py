"""BEV 编码器：查询当前图像后查询带实际变换几何的上一帧 BEV，再由 ConvNeXt2D 提炼。

模块: model/bev_encoder/bev_encoder.py
依赖: torch, config.schema.DrivingCfg, model.attention.CrossAttentionBlock,
      model.residual_block.ConvNeXtBlock2d, model.bev_encoder.checks.bev_encoder_checks
读取配置:
    model.driving.work_dim
    model.driving.attention.num_heads / mlp_ratio
    model.driving.bev_encoder.cross_layers / temporal_layers / num_convnext_blocks /
        convnext_spatial_kernel / convnext_expansion
对外接口:
    - BevEncoder(cfg_driving) -> nn.Module
        forward(bev_query, image_feat, previous_bev=None, previous_geometry=None,
                previous_valid=None) -> Tensor   # [B, work_dim, Hb, Wb]
说明: 把 BEV 查询网格展平为 Token，先以图像 patch 特征为 KV 级联查询；若提供历史帧，则再以
      `上一帧 BEV 骨干末端特征 + 刚性变换后实际 cell 坐标的共享几何编码`为 KV 级联查询。场景首帧通过
      previous_valid 在残差层面跳过时序结果，不使用全零伪历史。最后 reshape 回 BEV 网格，过
      num_convnext_blocks 层 ConvNeXt2D 在 BEV 空间维传播/提炼。输出为像素洗牌
      上采样前的 BEV 特征，同时供三场、独立道路线图与轨迹解码取用。精度由外层 autocast 控制。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config.schema import DrivingCfg
from model.attention import CrossAttentionBlock
from model.bev_encoder.checks.bev_encoder_checks import check_bev_encoder_inputs
from model.residual_block import ConvNeXtBlock2d


__all__ = ["BevEncoder"]


class BevEncoder(nn.Module):
    """交叉注意力 + ConvNeXt2D 的 BEV 编码器。

    Args:
        cfg_driving: 驾驶配置 `config.model.driving`。

    Shape:
        bev_query: `[B, work_dim, Hb, Wb]`，image_feat: `[B, work_dim, gh, gw]`；
        输出: `[B, work_dim, Hb, Wb]`。
    """

    def __init__(self, cfg_driving: DrivingCfg) -> None:
        super().__init__()
        self.work_dim = cfg_driving.work_dim
        attn = cfg_driving.attention
        be = cfg_driving.bev_encoder
        self.image_cross = nn.ModuleList(
            CrossAttentionBlock(self.work_dim, attn.num_heads, attn.mlp_ratio)
            for _ in range(be.cross_layers))
        self.temporal_cross = nn.ModuleList(
            CrossAttentionBlock(self.work_dim, attn.num_heads, attn.mlp_ratio)
            for _ in range(be.temporal_layers))
        self.convnext = nn.Sequential(*(
            ConvNeXtBlock2d(self.work_dim, be.convnext_spatial_kernel, be.convnext_expansion)
            for _ in range(be.num_convnext_blocks)))

    def forward(self, bev_query: torch.Tensor, image_feat: torch.Tensor,
                previous_bev: torch.Tensor = None, previous_geometry: torch.Tensor = None,
                previous_valid: torch.Tensor = None) -> torch.Tensor:
        """依次查询当前图像与可选历史 BEV，返回骨干末端特征 `[B, work_dim,Hb,Wb]`。"""
        check_bev_encoder_inputs(
            bev_query, image_feat, self.work_dim, previous_bev, previous_geometry, previous_valid)
        b, c, hb, wb = bev_query.shape
        query = bev_query.flatten(2).transpose(1, 2)   # [B, Hb*Wb, C]
        context = image_feat.flatten(2).transpose(1, 2)  # [B, gh*gw, C]
        for layer in self.image_cross:
            query = layer(query, context)
        if previous_bev is not None:
            history = (previous_bev + previous_geometry).flatten(2).transpose(1, 2)
            valid = previous_valid.to(query.dtype)[:, None, None]
            for layer in self.temporal_cross:
                fused = layer(query, history)
                query = query + valid * (fused - query)
        bev = query.transpose(1, 2).reshape(b, c, hb, wb)
        return self.convnext(bev)
