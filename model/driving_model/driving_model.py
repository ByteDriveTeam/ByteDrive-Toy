"""双帧开环驾驶模型：融合刚性对齐的历史 BEV，解码三场、道路线、交通控制与驾驶输出。

模块: model/driving_model/driving_model.py
依赖: torch, contextlib, config.schema.Config, model.perception_model.PerceptionFeatureEncoder,
      model.driving_neck.DrivingNeck, model.bev_query_embedding.BevQueryEmbedding,
      model.bev_encoder.BevEncoder, model.field_decoder.FieldDecoder,
      model.lane_map_decoder.LaneMapDecoder, model.trajectory_decoder.TrajectoryDecoder,
      model.driving_model.checks.driving_model_checks
读取配置:
    model.driving.work_dim / freeze_perception / neck_num_residual_blocks
    model.driving.bev / query / frustum / attention / bev_encoder / fields / lane_map /
        traffic_control / trajectory / behavior 各键
    model.dinov3_backbone.patch_size / hidden_dim（frustum 像素反投影、DINO 原始特征通道）
    model.feature_trunk.channels（trunk 通道）
对外接口:
    - DrivingModel(cfg) -> nn.Module
        forward(rgb, intrinsics, extrinsics, target_point, ego_velocity, previous_rgb,
                previous_to_current, previous_valid) -> dict
            # 三场 + 道路线/停止线/灯色 + trajectories/confidence/behavior_logits
        trainable_parameters() -> Iterator[nn.Parameter]   # 驾驶各件 + 可选感知 fusion/trunk
说明: Driving 仅构造不含像素头的 PerceptionFeatureEncoder，取得 trunk 末端特征与 DINOv3 原始特征；driving_neck
      融合并注入 frustum 几何位置编码得图像特征；bev_query_embedding 仅以 BEV xyz 几何初始化查询，
      bev_encoder 用交叉注意力聚合图像与历史，再以带无位置 BEV 寄存器的六层二维 RoPE Transformer 提炼；
      field_decoder 上采样解码三场。上一帧由同一套纯几何查询得到 BEV 骨干末端特征；其每个 cell 的坐标由
      previous_to_current 刚性变换到当前 ego 系，并通过与当前查询共享的几何编码器按真实变换坐标重编码。当前
      BEV 查询先查图像、再查上一帧 BEV。lane_map_decoder 输出道路线、停止线与灯色；trajectory_decoder
      以目标点、ego 平面速度为条件，用可学习 Mode Token 依次查询主干第 3/6 层。前向单目，自车位于 BEV 下方中心。
      混精边界（外置）：感知提特征 + neck + BEV 编码在 BF16 autocast 下；末端场上采样/解码与轨迹解码在 FP32。
      freeze_perception 为真时视觉编码器冻结且在 no_grad 下前向，梯度只回传驾驶各件；为假时优化驾驶实际经过的
      fusion/trunk。语义/深度解码头不构造、不挂入模型树，也不参与驾驶权重加载。
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict, Iterator

import torch
import torch.nn as nn

from config.schema import Config
from model.bev_encoder import BevEncoder
from model.bev_query_embedding import BevQueryEmbedding
from model.driving_model.checks.driving_model_checks import check_driving_inputs
from model.driving_neck import DrivingNeck
from model.field_decoder import FieldDecoder
from model.lane_map_decoder import LaneMapDecoder
from model.perception_model import PerceptionFeatureEncoder
from model.trajectory_decoder import TrajectoryDecoder


__all__ = ["DrivingModel"]

_LOW_PRECISION = torch.bfloat16


class DrivingModel(nn.Module):
    """双帧开环驾驶模型（复用感知主干并融合上一帧 BEV）。

    Args:
        cfg: 全局配置，读取 `model.driving` 及感知骨干/主干维度。

    Shape:
        当前/上一帧 rgb `[B,3,H,W]`、intrinsics `[B,4]`、extrinsics `[B,6]`、
        target_point/ego_velocity `[B,2]`、
        previous_to_current `[B,3,3]`、previous_valid `[B]`。
        输出 dict：risk/drivable/distribution/stop_line_logits、交通灯状态、道路线、轨迹与行为。
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        drv = cfg.model.driving
        bb = cfg.model.dinov3_backbone
        self.freeze_perception = drv.freeze_perception
        self.detach_previous = drv.bev_encoder.detach_previous

        self.perception = PerceptionFeatureEncoder(cfg)
        # DINOv3 骨干恒冻结；freeze_perception=True 时连同融合/trunk 冻结，否则以较小学习率适配驾驶任务。
        if self.freeze_perception:
            self.perception.requires_grad_(False)

        self.neck = DrivingNeck(drv, cfg.model.feature_trunk.channels, bb.hidden_dim, bb.patch_size)
        self.query = BevQueryEmbedding(
            out_dim=drv.work_dim,
            x_min_m=drv.bev.x_min_m, x_max_m=drv.bev.x_max_m,
            y_min_m=drv.bev.y_min_m, y_max_m=drv.bev.y_max_m,
            height=drv.bev.height, width=drv.bev.width,
            z_min_m=drv.bev.z_min_m, z_max_m=drv.bev.z_max_m, z_step_m=drv.bev.z_step_m,
            coord_symlog_scale=drv.query.coord_symlog_scale, mlp_hidden=drv.query.mlp_hidden)
        self.bev_encoder = BevEncoder(drv)
        self.field_decoder = FieldDecoder(drv)
        self.lane_map_decoder = LaneMapDecoder(drv)
        self.trajectory_decoder = TrajectoryDecoder(drv)

    def _driving_modules(self):
        """驾驶新增模块（不含复用的感知子模块）。"""
        return (self.neck, self.query, self.bev_encoder, self.field_decoder,
                self.lane_map_decoder, self.trajectory_decoder)

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """可训练参数：驾驶各件，以及未冻结时驾驶前向实际使用的感知 fusion/trunk。"""
        for m in self._driving_modules():
            yield from m.parameters()
        if not self.freeze_perception:
            yield from self.perception.feature_parameters()

    def param_groups(self, base_lr: float, weight_decay: float, perception_lr_scale: float):
        """优化器参数分组：驾驶各件用 base_lr；感知子模块（若未冻结）用 base_lr·perception_lr_scale 慢更新。

        DINOv3 骨干恒冻结、不出现在任何分组；fusion/trunk 以极小 lr 缓慢适配驾驶任务。驾驶模型不构造
        语义/深度头，因此模型树和优化器均不存在对应参数。返回 AdamW 参数组。
        """
        groups = [{"params": [p for m in self._driving_modules() for p in m.parameters()],
                   "lr": base_lr, "weight_decay": weight_decay}]
        if not self.freeze_perception:
            groups.append({"params": list(self.perception.feature_parameters()),
                           "lr": base_lr * perception_lr_scale, "weight_decay": weight_decay})
        return groups

    def forward(self, rgb: torch.Tensor, intrinsics: torch.Tensor, extrinsics: torch.Tensor,
                target_point: torch.Tensor, ego_velocity: torch.Tensor, previous_rgb: torch.Tensor,
                previous_to_current: torch.Tensor,
                previous_valid: torch.Tensor) -> Dict[str, torch.Tensor]:
        """双帧前向：当前图像 → 上一帧 BEV → 当前 BEV → 三场/道路线图/轨迹行为。"""
        check_driving_inputs(
            rgb, intrinsics, extrinsics, target_point, ego_velocity, previous_rgb,
            previous_to_current, previous_valid)
        device = rgb.device
        batch_size = int(rgb.shape[0])

        # BF16 段：当前帧先查询图像，随后查询携带变换后真实几何的上一帧 BEV。
        with self._autocast(device, enabled=True):
            image_feat = self._image_features(rgb, intrinsics, extrinsics)
            bev_query = self.query(batch_size, device)
            history_context = torch.no_grad() if self.detach_previous else nullcontext()
            with history_context:
                previous_image = self._image_features(previous_rgb, intrinsics, extrinsics)
                previous_bev = self.bev_encoder(
                    bev_query, previous_image)

            transformed_grid = self._transformed_previous_grid(previous_to_current)
            previous_geometry = self.query(batch_size, device, transformed_grid)
            bev_feat, planning_features = self.bev_encoder(
                bev_query, image_feat, previous_bev, previous_geometry, previous_valid,
                return_intermediate=True)

        # FP32 段：三场、独立道路线图与轨迹/行为末端解码。
        with self._autocast(device, enabled=False):
            bev_feat = bev_feat.float()
            outputs = self.field_decoder(bev_feat)
            outputs.update(self.lane_map_decoder(bev_feat))
            outputs.update(self.trajectory_decoder(
                tuple(feature.float() for feature in planning_features),
                target_point.float(), ego_velocity.float()))
        return outputs

    def _image_features(self, rgb, intrinsics, extrinsics):
        with torch.no_grad() if self.freeze_perception else nullcontext():
            trunk_feat, dino_raw = self.perception.extract_features(rgb)
        return self.neck(trunk_feat, dino_raw, intrinsics, extrinsics)

    def _transformed_previous_grid(self, previous_to_current):
        """把上一帧每个 BEV cell 中心刚性变换到当前 ego 系，保留真实几何供共享编码器使用。"""
        grid = self.query.grid_xy.float()
        grid_h = torch.cat((grid, torch.ones_like(grid[..., :1])), dim=-1)
        transformed = torch.einsum("bij,hwj->bhwi", previous_to_current.float(), grid_h)
        return transformed[..., :2]

    def _autocast(self, device: torch.device, enabled: bool) -> Any:
        """构造 autocast 上下文：enabled 用 BF16，否则关闭；meta/不支持设备回退空上下文。"""
        if device.type == "meta":
            return nullcontext()
        try:
            return torch.autocast(device_type=device.type, dtype=_LOW_PRECISION, enabled=enabled)
        except (RuntimeError, ValueError):
            return nullcontext()
