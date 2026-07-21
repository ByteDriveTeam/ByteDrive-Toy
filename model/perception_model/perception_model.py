"""共享视觉特征编码器，以及在其上追加语义/深度双头的多任务单帧感知模型。

模块: model/perception_model/perception_model.py
依赖: torch, contextlib, config.schema.Config, model.dinov3_backbone.DinoV3Backbone,
      model.feature_fusion.DinoFeatureFusion, model.feature_trunk.FeatureTrunk,
      model.perception_head.PerceptionHead, model.perception_model.checks.perception_model_checks
读取配置:
    model.dinov3_backbone（构造骨干；hidden_dim 与 feature_layers 数决定融合入口维）
    model.feature_trunk（channels / num_layers / num_heads / mlp_ratio / rope_theta）
    model.heads.reduce_channels / up_channels / semantic_out / depth_out
对外接口:
    - PerceptionFeatureEncoder(cfg) -> nn.Module   # 仅骨干+融合+trunk，供驾驶复用
        extract_features([B,3,H,W]) -> (trunk_feat, dino_raw)
    - PerceptionModel(cfg) -> nn.Module   # forward([B,3,H,W]) -> {"semantic","depth"}
    - trainable_parameters() -> Iterator[nn.Parameter]   # 仅 融合+trunk+双头（骨干冻结不训练）
    - feature_parameters() -> Iterator[nn.Parameter]     # 仅融合+trunk，供驾驶特征复用路径优化
说明: 单帧过冻结骨干得多层完整 Token 序列，经 feature_fusion 逐层 RMSNorm+拼接降到 channels，
      在保留 CLS/register/patch 顺序的前提下过三层 Pre-Norm Transformer，再仅取 patch
      还原网格。PerceptionFeatureEncoder 到此结束，不构造任何像素头；PerceptionModel 才追加双头并各自
      上采样至原分辨率。Transformer 的二维 RoPE 从 patch (1,1) 开始、步长 1，CLS/寄存器不编码。
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


__all__ = ["PerceptionFeatureEncoder", "PerceptionModel"]

# 骨干与主干段的低精度：任务要求骨干 BF16；末段上采样/解码/监督恒 FP32（设计常量，非实验参数）
_LOW_PRECISION = torch.bfloat16


class PerceptionFeatureEncoder(nn.Module):
    """仅含 DINOv3、特征融合与 Transformer trunk 的共享视觉编码器。

    Args:
        cfg: 全局配置，仅读取 `model.dinov3_backbone / feature_trunk`。

    Shape:
        输入: `[B, 3, H, W]`，H/W 为 patch_size 整数倍。
        extract_features 输出: trunk 与 DINO 原始 patch 网格特征。
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        bb = cfg.model.dinov3_backbone
        trunk_channels = cfg.model.feature_trunk.channels

        self.patch_size = bb.patch_size
        self.backbone = DinoV3Backbone(bb)
        # 融合入口维取自骨干：hidden_dim × 选层数；输出维 = 主干工作维，故融合即完成降维
        self.fusion = DinoFeatureFusion(bb.hidden_dim, len(bb.feature_layers), trunk_channels)
        self.trunk = FeatureTrunk(cfg.model.feature_trunk)

    def feature_parameters(self) -> Iterator[nn.Parameter]:
        """返回特征路径实际使用的融合层与 trunk 参数，骨干冻结不纳入。"""
        return (p for module in (self.fusion, self.trunk) for p in module.parameters())

    def extract_features(self, frames: torch.Tensor):
        """供下游驾驶系统复用的中段表征：返回 (trunk 末端特征, DINOv3 原始特征)。

        trunk 末端特征 `[B, channels, gh, gw]` 由三层 Transformer 输出的 patch Token 还原；
        DINOv3 原始特征取骨干末选层序列尾部 patch 网格 `[B, hidden, gh, gw]`。二者在 BF16 段计算，
        供驾驶 neck 各自 RMSNorm 后 1×1 融合（混精边界由本段 autocast 控制）。
        """
        check_input_frames(frames, self.patch_size)
        grid_height = int(frames.shape[2]) // self.patch_size
        grid_width = int(frames.shape[3]) // self.patch_size
        with self._autocast(frames.device, enabled=True):
            sequences = self.backbone(frames)  # [B,L,1+R+P,hidden]，冻结叶子
            dino_raw = _patch_grid(sequences[:, -1], grid_height, grid_width)
            trunk_tokens = self.trunk(
                self.fusion(sequences), grid_height, grid_width
            )
            trunk_feat = _patch_grid(trunk_tokens, grid_height, grid_width)
        return trunk_feat, dino_raw

    def _autocast(self, device: torch.device, enabled: bool) -> Any:
        """构造 autocast 上下文：enabled 时用 BF16，否则关闭；meta/不支持设备回退空上下文。"""
        if device.type == "meta":
            return nullcontext()
        try:
            return torch.autocast(device_type=device.type, dtype=_LOW_PRECISION, enabled=enabled)
        except (RuntimeError, ValueError):
            return nullcontext()


class PerceptionModel(PerceptionFeatureEncoder):
    """共享视觉编码器 + 语义/深度双头的多任务单帧感知模型。

    Args:
        cfg: 全局配置，读取 `model.dinov3_backbone / feature_trunk / heads`。

    Shape:
        输入: `[B, 3, H, W]`，H/W 为 patch_size 整数倍。
        输出: dict，`semantic [B,num_classes,H,W]`、`depth [B,2,H,W]`。
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        heads = cfg.model.heads
        trunk_channels = cfg.model.feature_trunk.channels
        self.semantic_head = PerceptionHead(
            trunk_channels, heads.reduce_channels, heads.up_channels, heads.semantic_out)
        self.depth_head = PerceptionHead(
            trunk_channels, heads.reduce_channels, heads.up_channels, heads.depth_out)

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """仅返回可训练参数（融合 + trunk + 双头）；骨干冻结不纳入优化器。"""
        modules = (self.fusion, self.trunk, self.semantic_head, self.depth_head)
        return (p for module in modules for p in module.parameters())

    def forward(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """单帧提特征 → 特征主干 → 双头上采样解码，返回双任务输出。"""
        check_input_frames(frames, self.patch_size)
        grid_height = int(frames.shape[2]) // self.patch_size
        grid_width = int(frames.shape[3]) // self.patch_size

        heads = {"semantic": self.semantic_head, "depth": self.depth_head}
        # BF16 段：骨干（no_grad 内部）+ 融合 + 主干 + 双头 encode
        with self._autocast(frames.device, enabled=True):
            sequences = self.backbone(frames)  # [B,L,1+R+P,hidden]，bf16 冻结叶子
            tokens = self.fusion(sequences)    # [B,1+R+P,channels]，Token 顺序不变
            tokens = self.trunk(tokens, grid_height, grid_width)
            feat = _patch_grid(tokens, grid_height, grid_width)
            encoded = {name: head.encode(feat) for name, head in heads.items()}

        # FP32 段：每头最后一次上采样 + 最终解码
        with self._autocast(frames.device, enabled=False):
            return {name: heads[name].decode(x.float()) for name, x in encoded.items()}


def _patch_grid(sequence: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """仅在进入像素解码路径时取序列尾部 patch Token，按 DINOv3 行主序还原网格。"""
    batch_size = int(sequence.shape[0])
    patch_count = height * width
    return (
        sequence[:, -patch_count:]
        .reshape(batch_size, height, width, -1)
        .permute(0, 3, 1, 2)
        .contiguous()
    )
