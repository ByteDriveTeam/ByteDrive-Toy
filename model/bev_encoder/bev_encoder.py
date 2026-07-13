"""BEV 编码器：初始 BEV 查询经级联交叉注意力查询图像特征，再过 ConvNeXt2D 提炼为 BEV 特征。

模块: model/bev_encoder/bev_encoder.py
依赖: torch, config.schema.DrivingCfg, model.attention.CrossAttentionBlock,
      model.residual_block.ConvNeXtBlock2d, model.bev_encoder.checks.bev_encoder_checks
读取配置:
    model.driving.work_dim
    model.driving.attention.num_heads / mlp_ratio
    model.driving.bev_encoder.cross_layers / num_convnext_blocks / convnext_spatial_kernel / convnext_expansion
对外接口:
    - BevEncoder(cfg_driving) -> nn.Module
        forward(bev_query, image_feat) -> Tensor   # [B, work_dim, Hb, Wb]
说明: 把 BEV 查询网格展平为 Token 作 query、图像 patch 特征展平为 KV，级联 cross_layers 层 Pre-Norm 交叉
      注意力，让每个 BEV cell 从图像相关区域聚合表观信息（几何对齐先验已由 frustum 注入图像侧特征）。注意力
      后 reshape 回 BEV 网格，过 num_convnext_blocks 层 ConvNeXt2D 在 BEV 空间维传播/提炼。输出为像素洗牌
      上采样前的 BEV 特征，同时供三场解码与轨迹解码取用。精度由外层 autocast 控制。
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
        self.cross = nn.ModuleList(
            CrossAttentionBlock(self.work_dim, attn.num_heads, attn.mlp_ratio)
            for _ in range(be.cross_layers))
        self.convnext = nn.Sequential(*(
            ConvNeXtBlock2d(self.work_dim, be.convnext_spatial_kernel, be.convnext_expansion)
            for _ in range(be.num_convnext_blocks)))

    def forward(self, bev_query: torch.Tensor, image_feat: torch.Tensor) -> torch.Tensor:
        """BEV 查询查询图像特征并提炼，返回 BEV 特征 `[B, work_dim, Hb, Wb]`。"""
        check_bev_encoder_inputs(bev_query, image_feat, self.work_dim)
        b, c, hb, wb = bev_query.shape
        query = bev_query.flatten(2).transpose(1, 2)   # [B, Hb*Wb, C]
        context = image_feat.flatten(2).transpose(1, 2)  # [B, gh*gw, C]
        for layer in self.cross:
            query = layer(query, context)
        bev = query.transpose(1, 2).reshape(b, c, hb, wb)
        return self.convnext(bev)
