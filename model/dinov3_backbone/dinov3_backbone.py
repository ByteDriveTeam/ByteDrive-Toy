"""DINOv3 ViT-S+ 视觉骨干：全程冻结 + eval，逐帧输出选定层的完整 Token 序列。

模块: model/dinov3_backbone/dinov3_backbone.py
依赖: torch, transformers(AutoModel), config.schema.DinoV3BackboneCfg, model.dinov3_backbone.checks.dinov3_backbone_checks
读取配置:
    model.dinov3_backbone.model_dir
    model.dinov3_backbone.patch_size
    model.dinov3_backbone.hidden_dim
    model.dinov3_backbone.num_register_tokens
    model.dinov3_backbone.feature_layers
对外接口:
    - DinoV3Backbone(cfg) -> nn.Module   # forward([N,3,H,W]) -> [N,L,1+R+P,hidden]
说明: 骨干是师生头共享的冻结特征源，参数一律 requires_grad=False 且恒 eval（覆写 train 使 .train()
      不解冻）；前向在 no_grad 下运行，返回的特征作为可训练头的叶子输入，梯度不回传骨干、省显存。
      经 output_hidden_states 取出 feature_layers 指定的多层（浅/中/深层，各 [N,seq,hidden]），
      保留每层原始的 1 CLS + register + patch 顺序并堆叠为 [N,L,seq,hidden]，交由下游
      feature_fusion 在 Token 维度不变的前提下融合；特殊 Token 直至预测头前都不被裁剪。
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
    check_sequence_tokens,
)


__all__ = ["DinoV3Backbone"]

# 仓库根：本文件位于 <root>/model/，故上跳一级即根，用于把相对 model_dir 解析为绝对路径
_REPO_ROOT = Path(__file__).resolve().parents[2]


class DinoV3Backbone(nn.Module):
    """冻结的 DINOv3 ViT-S+ 骨干。

    Args:
        cfg: 骨干配置，唯一来源为 `config.model.dinov3_backbone`。

    Shape:
        输入: `[N, 3, H, W]`，H/W 为 patch_size 整数倍。
        输出: `[N,L,1+num_register_tokens+H/patch_size·W/patch_size,hidden_dim]`。
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
        """逐帧提取选定层完整 Token 序列，按层堆叠为 [N,L,S,hidden]。"""
        check_backbone_frames(frames, self.cfg.patch_size)
        grid_height = int(frames.shape[2]) // self.cfg.patch_size
        grid_width = int(frames.shape[3]) // self.cfg.patch_size
        num_patches = grid_height * grid_width

        # output_hidden_states 返回 embedding(索引0)+各层输出，共 num_layers+1 个 [N,seq,hidden]
        hidden_states = self.model(pixel_values=frames, output_hidden_states=True).hidden_states
        check_feature_layers(len(hidden_states), self.cfg.feature_layers)
        check_sequence_tokens(
            int(hidden_states[0].shape[1]), num_patches, self.cfg.num_register_tokens)

        # 层维外的 CLS/register/patch 顺序完整继承 DINOv3，不在骨干边界裁剪。
        return torch.stack([hidden_states[i] for i in self.cfg.feature_layers], dim=1).contiguous()


def _load_frozen_dinov3(model_dir: str) -> nn.Module:
    """从本地目录加载 DINOv3 权重（local_files_only，不联网）。"""
    from transformers import AutoModel

    resolved = model_dir if Path(model_dir).is_absolute() else str(_REPO_ROOT / model_dir)
    return AutoModel.from_pretrained(resolved, local_files_only=True)
