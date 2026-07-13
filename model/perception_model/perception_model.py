"""多任务单帧感知模型：冻结 DINOv3 骨干 + 多层特征融合 + 2D 特征主干 + 语义/深度双头。

模块: model/perception_model/perception_model.py
依赖: torch, contextlib, config.schema.Config, model.dinov3_backbone.DinoV3Backbone,
      model.feature_fusion.DinoFeatureFusion, model.feature_trunk.FeatureTrunk,
      model.perception_head.PerceptionHead, model.perception_model.checks.perception_model_checks
读取配置:
    model.dinov3_backbone（构造骨干；hidden_dim 与 feature_layers 数决定融合入口维）
    model.feature_trunk.channels（= 融合输出维）
    model.heads.reduce_channels / up_channels / semantic_out / depth_out
对外接口:
    - PerceptionModel(cfg) -> nn.Module   # forward([B,3,H,W]) -> {"semantic","depth"}
    - trainable_parameters() -> Iterator[nn.Parameter]   # 仅 融合+trunk+双头（骨干冻结不训练）
说明: 单帧过冻结骨干得多层 patch 网格特征，经 feature_fusion 逐层 RMSNorm+拼接降到 channels，
      过 2D 特征主干后由双头各自上采样至原分辨率。
      精度边界（规范：混精外置）：骨干+融合+主干+双头的 encode 段在 BF16 autocast 下运行；每头最后一次上采样
      与最终解码（head.decode）在 FP32 下运行，损失亦在 FP32。autocast 设备类型由输入张量所在设备推导，
      故 CPU/GPU 均可跑。骨干在 no_grad 下前向、参数冻结，梯度只回传 融合、trunk 与双头。
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict, Iterator

import torch
import torch.nn as nn

from config.schema import Config
from model.dinov3_backbone import DinoV3Backbone
from model.feature_fusion import DinoFeatureFusion
from model.feature_trunk import FeatureTrunk
from model.perception_head import PerceptionHead
from model.perception_model.checks.perception_model_checks import check_input_frames


__all__ = ["PerceptionModel"]

# 骨干与主干段的低精度：任务要求骨干 BF16；末段上采样/解码/监督恒 FP32（设计常量，非实验参数）
_LOW_PRECISION = torch.bfloat16


class PerceptionModel(nn.Module):
    """DINOv3 + 2D 特征主干 + 双头的多任务单帧感知模型。

    Args:
        cfg: 全局配置，读取 `model.dinov3_backbone / feature_trunk / heads`。

    Shape:
        输入: `[B, 3, H, W]`，H/W 为 patch_size 整数倍。
        输出: dict，`semantic [B,num_classes,H,W]`、`depth [B,2,H,W]`。
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        bb = cfg.model.dinov3_backbone
        heads = cfg.model.heads
        trunk_channels = cfg.model.feature_trunk.channels

        self.patch_size = bb.patch_size
        self.backbone = DinoV3Backbone(bb)
        # 融合入口维取自骨干：hidden_dim × 选层数；输出维 = 主干工作维，故融合即完成降维
        self.fusion = DinoFeatureFusion(bb.hidden_dim, len(bb.feature_layers), trunk_channels)
        self.trunk = FeatureTrunk(cfg.model.feature_trunk)
        # 双头共用结构，仅输出通道不同
        self.semantic_head = PerceptionHead(
            trunk_channels, heads.reduce_channels, heads.up_channels, heads.semantic_out)
        self.depth_head = PerceptionHead(
            trunk_channels, heads.reduce_channels, heads.up_channels, heads.depth_out)

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """仅返回可训练参数（融合 + trunk + 双头）；骨干冻结不纳入优化器。"""
        modules = (self.fusion, self.trunk, self.semantic_head, self.depth_head)
        return (p for m in modules for p in m.parameters())

    def forward(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """单帧提特征 → 特征主干 → 双头上采样解码，返回双任务输出。"""
        check_input_frames(frames, self.patch_size)

        heads = {"semantic": self.semantic_head, "depth": self.depth_head}
        # BF16 段：骨干（no_grad 内部）+ 融合 + 主干 + 双头 encode
        with self._autocast(frames.device, enabled=True):
            feat = self.backbone(frames)  # [B, L, hidden, gh, gw]，bf16 冻结叶子
            feat = self.fusion(feat)      # [B, channels, gh, gw]，融合多层并降到工作维
            feat = self.trunk(feat)       # [B, channels, gh, gw]
            encoded = {name: head.encode(feat) for name, head in heads.items()}

        # FP32 段：每头最后一次上采样 + 最终解码
        with self._autocast(frames.device, enabled=False):
            return {name: heads[name].decode(x.float()) for name, x in encoded.items()}

    def _autocast(self, device: torch.device, enabled: bool) -> Any:
        """构造 autocast 上下文：enabled 时用 BF16，否则关闭；meta/不支持设备回退空上下文。"""
        if device.type == "meta":
            return nullcontext()
        try:
            return torch.autocast(device_type=device.type, dtype=_LOW_PRECISION, enabled=enabled)
        except (RuntimeError, ValueError):
            return nullcontext()
