"""DINOv3 ViT-B 视觉骨干：全程冻结 + eval，逐帧输出 patch 网格特征。

模块: model/dinov3_backbone.py
依赖: torch, transformers(AutoModel), config.schema.DinoV3BackboneCfg, model.dinov3_backbone_checks
读取配置:
    model.dinov3_backbone.model_dir
    model.dinov3_backbone.patch_size
    model.dinov3_backbone.hidden_dim
    model.dinov3_backbone.num_register_tokens
对外接口:
    - DinoV3Backbone(cfg) -> nn.Module   # forward([N,3,H,W]) -> [N, hidden, H/patch, W/patch]
说明: 骨干是师生头共享的冻结特征源，参数一律 requires_grad=False 且恒 eval（覆写 train 使 .train()
      不解冻）；前向在 no_grad 下运行，返回的特征作为可训练头的叶子输入，梯度不回传骨干、省显存。
      patch token 取序列末 H/patch×W/patch 个，跳过前部 1 个 CLS 与 register，故与二者排布顺序解耦。
      本地权重目录相对仓库根解析；加载走 local_files_only，不联网。
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from config.schema import DinoV3BackboneCfg
from model.dinov3_backbone_checks import (
    check_backbone_cfg,
    check_backbone_frames,
    check_patch_tokens,
)


__all__ = ["DinoV3Backbone"]

# 仓库根：本文件位于 <root>/model/，故上跳一级即根，用于把相对 model_dir 解析为绝对路径
_REPO_ROOT = Path(__file__).resolve().parents[1]


class DinoV3Backbone(nn.Module):
    """冻结的 DINOv3 ViT-B 骨干。

    Args:
        cfg: 骨干配置，唯一来源为 `config.model.dinov3_backbone`。

    Shape:
        输入: `[N, 3, H, W]`，H/W 为 patch_size 整数倍。
        输出: `[N, hidden_dim, H/patch_size, W/patch_size]`。
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
        """逐帧提取 patch 网格特征。"""
        check_backbone_frames(frames, self.cfg.patch_size)
        grid_height = int(frames.shape[2]) // self.cfg.patch_size
        grid_width = int(frames.shape[3]) // self.cfg.patch_size
        num_patches = grid_height * grid_width

        outputs = self.model(pixel_values=frames)
        sequence = outputs.last_hidden_state  # [N, 1+register+num_patches, hidden]
        check_patch_tokens(int(sequence.shape[1]), num_patches, self.cfg.num_register_tokens)

        patch_tokens = sequence[:, -num_patches:, :]  # 末 num_patches 个即 patch token
        # [N, num_patches, C] -> [N, gh, gw, C] -> [N, C, gh, gw]
        grid = patch_tokens.reshape(int(frames.shape[0]), grid_height, grid_width, -1)
        return grid.permute(0, 3, 1, 2).contiguous()


def _load_frozen_dinov3(model_dir: str) -> nn.Module:
    """从本地目录加载 DINOv3 权重（local_files_only，不联网）。"""
    from transformers import AutoModel

    resolved = model_dir if Path(model_dir).is_absolute() else str(_REPO_ROOT / model_dir)
    return AutoModel.from_pretrained(resolved, local_files_only=True)
