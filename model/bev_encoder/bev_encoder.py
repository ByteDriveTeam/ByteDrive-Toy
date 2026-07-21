"""BEV 编码器：融合当前图像与历史 BEV，再由带无位置寄存器的六层二维 RoPE Transformer 提炼。

模块: model/bev_encoder/bev_encoder.py
依赖: torch, config.schema.DrivingCfg, model.attention.CrossAttentionBlock,
      model.attention.ImageSelfAttentionBlock, model.bev_encoder.checks.bev_encoder_checks
读取配置:
    model.driving.work_dim
    model.driving.attention.num_heads / mlp_ratio
    model.driving.bev_encoder.cross_layers / temporal_layers / transformer_layers /
        num_register_tokens / register_init_std / rope_theta
对外接口:
    - BevEncoder(cfg_driving) -> nn.Module
        forward(bev_query, image_feat, previous_bev=None, previous_geometry=None,
                previous_valid=None, return_intermediate=False) -> Tensor | (Tensor, tuple[Tensor, Tensor])
            # 默认返回末层；规划路径可同时取得第 3、6 层 [B, work_dim, Hb, Wb]
说明: 把 BEV 查询网格展平为 Token，先以图像 patch 特征为 KV 级联查询；若提供历史帧，则再以
      `上一帧 BEV 骨干末端特征 + 刚性变换后实际 cell 坐标的共享几何编码`为 KV 级联查询。场景首帧通过
      previous_valid 在残差层面跳过时序结果，不使用全零伪历史。随后在 BEV Patch 前拼接本编码器自己的
      可学习寄存器 Token，共同经过六层 Pre-Norm Transformer；二维 RoPE 仅编码从 (1,1) 起的 BEV Patch，
      寄存器无位置并用于吸纳噪声，输出前丢弃。精度由外层 autocast 控制。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config.schema import DrivingCfg
from model.attention import CrossAttentionBlock, ImageSelfAttentionBlock
from model.bev_encoder.checks.bev_encoder_checks import check_bev_encoder_inputs


__all__ = ["BevEncoder"]


class BevEncoder(nn.Module):
    """交叉注意力 + 带无位置寄存器的二维 RoPE Transformer BEV 编码器。

    参数:
        cfg_driving: 驾驶配置 `config.model.driving`。

    形状:
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
        self.register_tokens = nn.Parameter(torch.zeros(1, be.num_register_tokens, self.work_dim))
        nn.init.normal_(self.register_tokens, std=be.register_init_std)
        self.transformer = nn.ModuleList(
            ImageSelfAttentionBlock(
                self.work_dim, attn.num_heads, attn.mlp_ratio, be.rope_theta)
            for _ in range(be.transformer_layers))

    def forward(self, bev_query: torch.Tensor, image_feat: torch.Tensor,
                previous_bev: torch.Tensor = None, previous_geometry: torch.Tensor = None,
                previous_valid: torch.Tensor = None, return_intermediate: bool = False):
        """依次查询当前图像与历史 BEV；可选同时返回主干第 3、6 层特征供规划查询。"""
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
        registers = self.register_tokens.expand(b, -1, -1).to(query.dtype)
        sequence = torch.cat((registers, query), dim=1)
        patch_positions = _patch_positions(hb, wb, query.device)
        intermediate = []
        for layer_index, layer in enumerate(self.transformer, start=1):
            sequence = layer(sequence, patch_positions)
            if return_intermediate and layer_index in (3, 6):
                intermediate.append(self._patch_map(sequence, b, c, hb, wb))
        output = self._patch_map(sequence, b, c, hb, wb)
        return (output, tuple(intermediate)) if return_intermediate else output

    def _patch_map(self, sequence, batch_size, channels, height, width):
        """丢弃无位置寄存器并恢复 `[B,C,H,W]` Patch 特征图。"""
        patches = sequence[:, self.register_tokens.shape[1]:]
        return patches.transpose(1, 2).reshape(batch_size, channels, height, width)


def _patch_positions(height: int, width: int, device: torch.device) -> torch.Tensor:
    """按 BEV 展平顺序生成二维 RoPE 坐标，首个 Patch 为 (1,1)。"""
    rows = torch.arange(1, height + 1, device=device, dtype=torch.float32)
    columns = torch.arange(1, width + 1, device=device, dtype=torch.float32)
    row_grid, column_grid = torch.meshgrid(rows, columns, indexing="ij")
    return torch.stack((row_grid, column_grid), dim=-1).reshape(-1, 2)
