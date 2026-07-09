"""多任务时序感知模型：冻结 DINOv3 骨干 + 3D 时序主干 + 语义/光流/深度三头。

模块: model/perception_model.py
依赖: torch, contextlib, config.schema.Config, model.dinov3_backbone.DinoV3Backbone,
      model.temporal_trunk.TemporalTrunk, model.perception_head.PerceptionHead, model.perception_model_checks
读取配置:
    model.dinov3_backbone（构造骨干）
    model.temporal_trunk.channels
    model.heads.reduce_channels / up_channels / semantic_out / flow_out / depth_out
对外接口:
    - PerceptionModel(cfg) -> nn.Module   # forward([B,T,3,H,W]) -> {"semantic","flow","depth"}
    - trainable_parameters() -> Iterator[nn.Parameter]   # 仅 trunk + 三头（骨干冻结不训练）
说明: 逐帧过冻结骨干得 patch 网格特征，堆成 [B,C,T,gh,gw] 时序后过主干，再由三头各自上采样至原分辨率。
      精度边界（规范：混精外置）：骨干+主干+三头的 encode 段在 BF16 autocast 下运行；每头最后一次上采样
      与最终解码（head.decode）在 FP32 下运行，损失亦在 FP32。autocast 设备类型由输入张量所在设备推导，
      故 CPU/GPU 均可跑。骨干在 no_grad 下前向、参数冻结，梯度只回传 trunk 与三头。
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict, Iterator

import torch
import torch.nn as nn

from config.schema import Config
from model.dinov3_backbone import DinoV3Backbone
from model.perception_head import PerceptionHead
from model.perception_model_checks import check_input_frames
from model.temporal_trunk import TemporalTrunk


__all__ = ["PerceptionModel"]

# 骨干与主干段的低精度：任务要求骨干 BF16；末段上采样/解码/监督恒 FP32（设计常量，非实验参数）
_LOW_PRECISION = torch.bfloat16


class PerceptionModel(nn.Module):
    """DINOv3 + 3D ConvNeXt 主干 + 三头的多任务感知模型。

    Args:
        cfg: 全局配置，读取 `model.dinov3_backbone / temporal_trunk / heads`。

    Shape:
        输入: `[B, T, 3, H, W]`，H/W 为 patch_size 整数倍。
        输出: dict，`semantic [B,num_classes,T,H,W]`、`flow [B,2,T,H,W]`、`depth [B,2,T,H,W]`。
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        bb = cfg.model.dinov3_backbone
        heads = cfg.model.heads
        trunk_channels = cfg.model.temporal_trunk.channels

        self.patch_size = bb.patch_size
        self.backbone = DinoV3Backbone(bb)
        self.trunk = TemporalTrunk(cfg.model.temporal_trunk)
        # 三头共用结构，仅输出通道不同
        self.semantic_head = PerceptionHead(
            trunk_channels, heads.reduce_channels, heads.up_channels, heads.semantic_out)
        self.flow_head = PerceptionHead(
            trunk_channels, heads.reduce_channels, heads.up_channels, heads.flow_out)
        self.depth_head = PerceptionHead(
            trunk_channels, heads.reduce_channels, heads.up_channels, heads.depth_out)

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """仅返回可训练参数（trunk + 三头）；骨干冻结不纳入优化器。"""
        modules = (self.trunk, self.semantic_head, self.flow_head, self.depth_head)
        return (p for m in modules for p in m.parameters())

    def forward(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """逐帧提特征 → 时序主干 → 三头上采样解码，返回三任务输出。"""
        check_input_frames(frames, self.patch_size)
        batch, time = int(frames.shape[0]), int(frames.shape[1])
        height, width = int(frames.shape[3]), int(frames.shape[4])
        grid_h, grid_w = height // self.patch_size, width // self.patch_size

        heads = {"semantic": self.semantic_head, "flow": self.flow_head, "depth": self.depth_head}
        # BF16 段：骨干（no_grad 内部）+ 主干 + 三头 encode
        with self._autocast(frames.device, enabled=True):
            flat = frames.reshape(batch * time, 3, height, width)
            feat = self.backbone(flat)  # [B*T, C, gh, gw]，bf16 冻结叶子
            channels = int(feat.shape[1])
            # [B*T,C,gh,gw] -> [B,T,C,gh,gw] -> [B,C,T,gh,gw]
            feat = feat.reshape(batch, time, channels, grid_h, grid_w).permute(0, 2, 1, 3, 4).contiguous()
            feat = self.trunk(feat)
            encoded = {name: head.encode(feat) for name, head in heads.items()}

        # FP32 段：每头最后一次上采样 + 最终解码
        with self._autocast(frames.device, enabled=False):
            return {name: heads[name].decode(x.float(), meta)
                    for name, (x, meta) in encoded.items()}

    def _autocast(self, device: torch.device, enabled: bool) -> Any:
        """构造 autocast 上下文：enabled 时用 BF16，否则关闭；meta/不支持设备回退空上下文。"""
        if device.type == "meta":
            return nullcontext()
        try:
            return torch.autocast(device_type=device.type, dtype=_LOW_PRECISION, enabled=enabled)
        except (RuntimeError, ValueError):
            return nullcontext()
