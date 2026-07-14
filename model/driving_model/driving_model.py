"""单帧开环驾驶模型：复用感知主干 → BEV → 三场 + 多模态轨迹与多标签行为。

模块: model/driving_model/driving_model.py
依赖: torch, contextlib, config.schema.Config, model.perception_model.PerceptionModel,
      model.driving_neck.DrivingNeck, model.target_point_embedding.TargetPointEmbedding,
      model.bev_encoder.BevEncoder, model.field_decoder.FieldDecoder,
      model.trajectory_decoder.TrajectoryDecoder, model.driving_model.checks.driving_model_checks
读取配置:
    model.driving.work_dim / freeze_perception 及其下 bev / query 各键
    model.dinov3_backbone.patch_size / hidden_dim（frustum 像素反投影、DINO 原始特征通道）
    model.feature_trunk.channels（trunk 通道）
对外接口:
    - DrivingModel(cfg) -> nn.Module
        forward(rgb, intrinsics, extrinsics, ego_velocity, target_point) -> dict
            # 三场 [B,1,Hf,Wf] + trajectories [B,M,T,2] + confidence [B,M] + behavior_logits [B,8]
        trainable_parameters() -> Iterator[nn.Parameter]   # 排除冻结感知主干
说明: 感知 = 驾驶的前端子任务：PerceptionModel 提供「双头共享 trunk 末端特征 + DINOv3 原始特征」，driving_neck
      融合并注入 frustum 几何位置编码得图像特征；target_point_embedding 由 ego 目标点生成初始 BEV 查询，
      bev_encoder 用交叉注意力从图像特征聚合信息并 ConvNeXt 提炼为 BEV 特征；field_decoder 上采样解码三场，
      trajectory_decoder 取（上采样前的）BEV 特征并入自车速度，以同一 Token 序列联合解码多模态轨迹与多标签
      行为。前向单目，自车位于 BEV 下方中心。
      混精边界（外置）：感知提特征 + neck + BEV 编码在 BF16 autocast 下；末端场上采样/解码与轨迹解码在 FP32。
      freeze_perception 为真时感知主干冻结且在 no_grad 下前向，梯度只回传驾驶各件（复用其深度/分割预训练表征）。
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict, Iterator

import torch
import torch.nn as nn

from config.schema import Config
from model.bev_encoder import BevEncoder
from model.driving_model.checks.driving_model_checks import check_driving_inputs
from model.driving_neck import DrivingNeck
from model.field_decoder import FieldDecoder
from model.perception_model import PerceptionModel
from model.target_point_embedding import TargetPointEmbedding
from model.trajectory_decoder import TrajectoryDecoder


__all__ = ["DrivingModel"]

_LOW_PRECISION = torch.bfloat16


class DrivingModel(nn.Module):
    """单帧开环驾驶模型（复用冻结感知主干）。

    Args:
        cfg: 全局配置，读取 `model.driving` 及感知骨干/主干维度。

    Shape:
        rgb `[B,3,H,W]`、intrinsics `[B,4]`、extrinsics `[B,6]`、ego_velocity `[B,2]`、target_point `[B,2]`。
        输出 dict：risk/drivable/distribution `[B,1,Hf,Wf]`、trajectories `[B,M,T,2]`、confidence `[B,M]`、
        behavior_logits `[B,8]`。
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        drv = cfg.model.driving
        bb = cfg.model.dinov3_backbone
        self.freeze_perception = drv.freeze_perception

        self.perception = PerceptionModel(cfg)
        # DINOv3 骨干恒冻结（PerceptionModel 内部保证）；freeze_perception=True 时连同融合/trunk/双头一并冻结，
        # 否则它们保持可训练、由优化器以 train.perception_lr_scale 慢更新（复用其深度/分割预训练表征）。
        if self.freeze_perception:
            self.perception.requires_grad_(False)

        self.neck = DrivingNeck(drv, cfg.model.feature_trunk.channels, bb.hidden_dim, bb.patch_size)
        self.query = TargetPointEmbedding(
            out_dim=drv.work_dim,
            x_min_m=drv.bev.x_min_m, x_max_m=drv.bev.x_max_m,
            y_min_m=drv.bev.y_min_m, y_max_m=drv.bev.y_max_m,
            height=drv.bev.height, width=drv.bev.width,
            z_min_m=drv.bev.z_min_m, z_max_m=drv.bev.z_max_m, z_step_m=drv.bev.z_step_m,
            coord_symlog_scale=drv.query.coord_symlog_scale, mlp_hidden=drv.query.mlp_hidden,
            vector_order=drv.query.vector_order)
        self.bev_encoder = BevEncoder(drv)
        self.field_decoder = FieldDecoder(drv)
        self.trajectory_decoder = TrajectoryDecoder(drv)

    def _driving_modules(self):
        """驾驶新增模块（不含复用的感知子模块）。"""
        return (self.neck, self.query, self.bev_encoder, self.field_decoder, self.trajectory_decoder)

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """可训练参数：驾驶各件；感知子模块视 freeze_perception 决定是否纳入。"""
        for m in self._driving_modules():
            yield from m.parameters()
        if not self.freeze_perception:
            yield from self.perception.trainable_parameters()

    def param_groups(self, base_lr: float, weight_decay: float, perception_lr_scale: float):
        """优化器参数分组：驾驶各件用 base_lr；感知子模块（若未冻结）用 base_lr·perception_lr_scale 慢更新。

        DINOv3 骨干恒冻结、不出现在任何分组；感知融合/trunk/双头以极小 lr 缓慢适配驾驶任务，既复用其预训练
        表征又允许轻微迁移。返回 AdamW 可直接消费的分组列表。
        """
        groups = [{"params": [p for m in self._driving_modules() for p in m.parameters()],
                   "lr": base_lr, "weight_decay": weight_decay}]
        if not self.freeze_perception:
            groups.append({"params": list(self.perception.trainable_parameters()),
                           "lr": base_lr * perception_lr_scale, "weight_decay": weight_decay})
        return groups

    def forward(self, rgb: torch.Tensor, intrinsics: torch.Tensor, extrinsics: torch.Tensor,
                ego_velocity: torch.Tensor, target_point: torch.Tensor) -> Dict[str, torch.Tensor]:
        """单帧前向：图像特征 → BEV → 三场 + 多模态轨迹/行为。"""
        check_driving_inputs(rgb, intrinsics, extrinsics, ego_velocity, target_point)
        device = rgb.device

        # BF16 段：感知提特征 + neck + 初始查询 + BEV 编码。完全冻结时 no_grad 省显存；慢更新时保留计算图
        # 使梯度回传融合/trunk/双头（DINOv3 骨干在 PerceptionModel 内部恒 no_grad，无论此处开关）。
        with self._autocast(device, enabled=True):
            with torch.no_grad() if self.freeze_perception else nullcontext():
                trunk_feat, dino_raw = self.perception.extract_features(rgb)
            image_feat = self.neck(trunk_feat, dino_raw, intrinsics, extrinsics)
            bev_query = self.query(target_point)
            bev_feat = self.bev_encoder(bev_query, image_feat)

        # FP32 段：场上采样/解码 + 轨迹解码（末端精度敏感）
        with self._autocast(device, enabled=False):
            bev_feat = bev_feat.float()
            outputs = self.field_decoder(bev_feat)
            outputs.update(self.trajectory_decoder(bev_feat, ego_velocity.float()))
        return outputs

    def _autocast(self, device: torch.device, enabled: bool) -> Any:
        """构造 autocast 上下文：enabled 用 BF16，否则关闭；meta/不支持设备回退空上下文。"""
        if device.type == "meta":
            return nullcontext()
        try:
            return torch.autocast(device_type=device.type, dtype=_LOW_PRECISION, enabled=enabled)
        except (RuntimeError, ValueError):
            return nullcontext()
