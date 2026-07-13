"""DINOv3 ViT-B 视觉骨干：全程冻结 + eval，逐帧输出多层 patch 网格特征。

模块: model/dinov3_backbone/dinov3_backbone.py
依赖: torch, transformers(AutoModel), config.schema.DinoV3BackboneCfg, model.dinov3_backbone.checks.dinov3_backbone_checks
读取配置:
    model.dinov3_backbone.model_dir
    model.dinov3_backbone.patch_size
    model.dinov3_backbone.hidden_dim
    model.dinov3_backbone.num_register_tokens
    model.dinov3_backbone.feature_layers
对外接口:
    - DinoV3Backbone(cfg) -> nn.Module   # forward([N,3,H,W]) -> [N, L, hidden, H/patch, W/patch]
说明: 骨干是师生头共享的冻结特征源，参数一律 requires_grad=False 且恒 eval（覆写 train 使 .train()
      不解冻）；前向在 no_grad 下运行，返回的特征作为可训练头的叶子输入，梯度不回传骨干、省显存。
      经 output_hidden_states 取出 feature_layers 指定的多层（浅/中/深层，各 [N,seq,hidden]），逐层
      切出 patch 网格后按层堆叠为 [N,L,hidden,gh,gw]，交由下游 feature_fusion 融合；单层维度不足以覆盖
      浅层纹理与深层语义，故多层并取。patch token 取序列末 gh×gw 个，跳过前部 1 CLS + register，与其排布解耦。
      本地权重目录相对仓库根解析；加载走 local_files_only，不联网。
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from config.schema import DinoV3BackboneCfg
from model.dinov3_backbone.checks.dinov3_backbone_checks import (
    check_backbone_cfg,
    check_backbone_frames,
    check_feature_layers,
    check_patch_tokens,
)


__all__ = ["DinoV3Backbone"]

# 仓库根：本文件位于 <root>/model/，故上跳一级即根，用于把相对 model_dir 解析为绝对路径
_REPO_ROOT = Path(__file__).resolve().parents[2]


class DinoV3Backbone(nn.Module):
    """冻结的 DINOv3 ViT-B 骨干。

    Args:
        cfg: 骨干配置，唯一来源为 `config.model.dinov3_backbone`。

    Shape:
        输入: `[N, 3, H, W]`，H/W 为 patch_size 整数倍。
        输出: `[N, L, hidden_dim, H/patch_size, W/patch_size]`，L = len(cfg.feature_layers)。
    """

    def __init__(self, cfg: DinoV3BackboneCfg) -> None:
        super().__init__()
        check_backbone_cfg(cfg)
        self.cfg = cfg
        self.model = _load_frozen_dinov3(cfg.model_dir)
        self.model.eval()
        self.model.requires_grad_(False)

    def train(self, mode: bool = True) -> "DinoV3Backbone":
        """覆写 train：骨干恒处 eval，避免外层 .train() 解冻或改变归一化统计。"""
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """逐帧提取多层 patch 网格特征，按层堆叠为 [N,L,hidden,gh,gw]。"""
        check_backbone_frames(frames, self.cfg.patch_size)
        num_frames = int(frames.shape[0])
        grid_height = int(frames.shape[2]) // self.cfg.patch_size
        grid_width = int(frames.shape[3]) // self.cfg.patch_size
        num_patches = grid_height * grid_width

        # output_hidden_states 返回 embedding(索引0)+各层输出，共 num_layers+1 个 [N,seq,hidden]
        hidden_states = self.model(pixel_values=frames, output_hidden_states=True).hidden_states
        check_feature_layers(len(hidden_states), self.cfg.feature_layers)
        check_patch_tokens(int(hidden_states[0].shape[1]), num_patches, self.cfg.num_register_tokens)

        # 逐层取末 num_patches 个 patch token 并还原为 [N, hidden, gh, gw]，再按层堆叠成 L 维
        grids = [
            hidden_states[i][:, -num_patches:, :]
            .reshape(num_frames, grid_height, grid_width, -1)
            .permute(0, 3, 1, 2)
            for i in self.cfg.feature_layers
        ]
        return torch.stack(grids, dim=1).contiguous()


def _load_frozen_dinov3(model_dir: str) -> nn.Module:
    """从本地目录加载 DINOv3 权重（local_files_only，不联网）。"""
    from transformers import AutoModel

    resolved = model_dir if Path(model_dir).is_absolute() else str(_REPO_ROOT / model_dir)
    return AutoModel.from_pretrained(resolved, local_files_only=True)
