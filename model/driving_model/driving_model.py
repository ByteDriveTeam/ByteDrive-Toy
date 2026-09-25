"""五帧三目图像与当前 LiDAR 的驾驶模型，输出独立场景、Agent 与规划任务。

模块: model/driving_model/driving_model.py
依赖: torch, config.schema, model.perception_model, model.driving_neck,
      model.driving_transformer, model.lidar_fusion, model.bev_decoder,
      model.trajectory_decoder, model.driving_model.checks
读取配置: model.driving.*, model.dinov3_backbone.patch_size/hidden_dim,
          model.feature_trunk.channels
对外接口:
    - DrivingModel(cfg) -> nn.Module
说明: 历史图像射线以四维刚性矩阵映射至当前 ego；位置编码只用于注意力 Q/K。
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.schema import Config
from model.bev_decoder import BevDecoder
from model.driving_model.checks.driving_model_checks import check_driving_inputs
from model.driving_neck import DrivingNeck
from model.driving_transformer import DrivingTransformer
from model.lidar_fusion import LidarQueryFusion
from model.perception_model import PerceptionFeatureEncoder
from model.trajectory_decoder import TrajectoryDecoder


__all__ = ["DrivingModel"]


class _LidarStep:
    """只修改 BEV Patch，Detect Token 保留独立内容。"""

    def __init__(self, fusion, stats, occupied, valid, bev_shape):
        self.fusion, self.stats = fusion, stats
        self.occupied, self.valid, self.bev_shape = occupied, valid, bev_shape

    def __call__(self, tokens):
        b, _, d = tokens.shape
        n = self.bev_shape[0] * self.bev_shape[1]
        bev = tokens[:, :n].transpose(1, 2).reshape(b, d, *self.bev_shape)
        fused = self.fusion(bev, bev, self.stats, self.occupied, self.valid)
        return torch.cat((fused.flatten(2).transpose(1, 2), tokens[:, n:]), dim=1)


class DrivingModel(nn.Module):
    """五帧图像 CA 与 1/3/5 层 LiDAR 融合的多任务模型。"""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        drv, bb = cfg.model.driving, cfg.model.dinov3_backbone
        self.freeze_perception = drv.freeze_perception
        self.perception = PerceptionFeatureEncoder(cfg)
        if self.freeze_perception:
            self.perception.requires_grad_(False)
        self.neck = DrivingNeck(drv, cfg.model.feature_trunk.channels,
                                bb.hidden_dim, bb.patch_size)
        self.encoder = DrivingTransformer(drv)
        self.lidar_fusers = nn.ModuleList(LidarQueryFusion(drv) for _ in range(3))
        self.decoder = BevDecoder(drv)
        self.trajectory_decoder = TrajectoryDecoder(drv)
        det = drv.detection
        self.detect_class = nn.ModuleList(nn.Linear(drv.work_dim, det.num_classes + 1)
                                          for _ in range(6))
        self.detect_box = nn.ModuleList(nn.Linear(drv.work_dim, 9) for _ in range(6))
        self.detect_future = nn.ModuleList(
            nn.Linear(drv.work_dim, det.num_modes * det.future_steps * 2)
            for _ in range(6))
        self.detect_modes, self.detect_steps = det.num_modes, det.future_steps
        bev = drv.bev
        self.register_buffer("box_scale", torch.tensor((bev.x_max_m - bev.x_min_m,
                         bev.y_max_m - bev.y_min_m, bev.z_max_m - bev.z_min_m)))

    def _driving_modules(self):
        return (self.neck, self.encoder, self.lidar_fusers, self.decoder,
                self.trajectory_decoder, self.detect_class, self.detect_box,
                self.detect_future)

    def trainable_parameters(self):
        """返回驾驶参数及可训练的感知融合主干参数。"""
        for module in self._driving_modules():
            yield from module.parameters()
        if not self.freeze_perception:
            yield from self.perception.feature_parameters()

    def param_groups(self, base_lr, weight_decay, perception_lr_scale):
        """感知路径使用配置指定的学习率倍率。"""
        groups = [{"params": [p for module in self._driving_modules()
                              for p in module.parameters()],
                   "lr": base_lr, "weight_decay": weight_decay}]
        if not self.freeze_perception:
            groups.append({"params": list(self.perception.feature_parameters()),
                           "lr": base_lr * perception_lr_scale,
                           "weight_decay": weight_decay})
        return groups

    def forward(self, rgb, intrinsics, extrinsics, target_point, ego_velocity,
                history_rgb, history_to_current, history_valid,
                lidar_stats=None, lidar_occupied=None, lidar_valid=None,
                trajectory=None, traj_valid=None, flow_time=None, flow_noise=None):
        """五帧特征一次编码，按层输出 Detect，并独立解码占用与轨迹。"""
        check_driving_inputs(rgb, intrinsics, extrinsics, target_point, ego_velocity,
                             history_rgb, history_to_current, history_valid)
        b = rgb.shape[0]
        frames = torch.cat((history_rgb, rgb[:, None]), dim=1)
        with torch.autocast(rgb.device.type, dtype=torch.bfloat16,
                            enabled=rgb.device.type in ("cuda", "cpu")):
            image, rays = self._image_features(frames, intrinsics, extrinsics)
            with torch.autocast(rgb.device.type, enabled=False):
                patches = rays.shape[1] // 15
                xyz = rays.reshape(b, 5, 3, patches, 5, rays.shape[-2], 3)
                rotation = history_to_current[..., :3, :3]
                translation = history_to_current[..., :3, 3]
                past = torch.einsum("bfij,bfvpqnj->bfvpqni", rotation.float(),
                                    xyz[:, :4].float()) + translation[:, :, None, None, None, None]
                rays = torch.cat((past, xyz[:, 4:]), 1).reshape(b, -1, 5, rays.shape[-2], 3)
            dt = self.cfg.model.driving.trajectory.waypoint_dt_s
            frame_time = torch.arange(-4, 1, device=rgb.device, dtype=torch.float32) * dt
            time = frame_time[None, :, None].expand(b, -1, 3 * patches).reshape(b, -1)
            valid = torch.cat((history_valid.bool(),
                               torch.ones(b, 1, device=rgb.device, dtype=torch.bool)), 1)
            valid = valid[:, :, None].expand(-1, -1, 3 * patches).reshape(b, -1)
            steps = tuple(_LidarStep(f, lidar_stats, lidar_occupied, lidar_valid,
                                     self.encoder.bev_shape) for f in self.lidar_fusers)
            bev, layers, detections, anchors = self.encoder(
                image, rays, time, valid, steps)
        outputs = self.decoder(bev.float())
        classes, boxes, futures = [], [], []
        for tokens, cls, box, future in zip(detections, self.detect_class,
                                             self.detect_box, self.detect_future):
            token = tokens.float()
            raw = box(token)
            center = anchors + torch.tanh(raw[..., :3]) * self.box_scale
            boxes.append(torch.cat((center, F.softplus(raw[..., 3:6]), raw[..., 6:]), -1))
            classes.append(cls(token))
            futures.append(future(token).reshape(b, -1, self.detect_modes,
                                                 self.detect_steps, 2))
        outputs.update({"detect_class_logits": torch.stack(classes),
                        "detect_boxes": torch.stack(boxes),
                        "detect_future": torch.stack(futures),
                        "detect_anchors": anchors})
        perception_tokens = tuple(torch.cat((
            layer.flatten(2).transpose(1, 2), detect_tokens), dim=1).float()
            for layer, detect_tokens in zip(layers, detections))
        outputs.update(self.trajectory_decoder(
            perception_tokens, anchors.float(), target_point.float(),
            ego_velocity.float(), trajectory=trajectory, traj_valid=traj_valid,
            flow_time=flow_time, flow_noise=flow_noise))
        return outputs

    def _image_features(self, frames, intrinsics, extrinsics):
        b, f, v, c, height, width = frames.shape
        rgb = frames.reshape(b * f * v, c, height, width)
        intr = intrinsics[:, None].expand(-1, f, -1, -1).reshape(-1, 4)
        extr = extrinsics[:, None].expand(-1, f, -1, -1).reshape(-1, 6)
        with torch.no_grad() if self.freeze_perception else nullcontext():
            trunk, dino = self.perception.extract_features(rgb)
        feature = self.neck(trunk, dino, intr, extr)
        gh, gw = feature.shape[-2:]
        with torch.autocast(rgb.device.type, enabled=False):
            rays = self.neck.frustum.ego_frustum_coords(gh, gw, intr, extr)
        return feature.flatten(2).transpose(1, 2).reshape(b, -1, feature.shape[1]), \
            rays.reshape(b, -1, 5, rays.shape[-2], 3)
