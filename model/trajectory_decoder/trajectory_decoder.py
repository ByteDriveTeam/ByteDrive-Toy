"""条件化多 Mode 规划解码器：以 8 个可学习 Token 查询主感知第 3/6 层特征并回归基线残差。

模块: model/trajectory_decoder/trajectory_decoder.py
依赖: torch, config.schema.DrivingCfg, data.target_encoding, model.attention,
      model.trajectory_decoder.checks.trajectory_decoder_checks
读取配置:
    model.driving.work_dim
    model.driving.bev.fov_deg
    model.driving.trajectory.num_modes / num_waypoints / planning_dim / condition_mlp_hidden /
        feature_ffn_hidden / cross_layers / self_layers / num_heads / mode_token_init_std /
        baseline_step_m / symlog_scale
    model.driving.behavior.num_classes
对外接口:
    - TrajectoryDecoder(cfg_driving) -> nn.Module
        forward(perception_features, target_point, ego_velocity) -> dict
            # trajectories/trajectory_residuals [B,M,T,2] / confidence [B,M] /
            # behavior_logits [B,C_behavior]
说明: 目标点与 ego 平面速度先作 Symlog（仅用于把量纲差异大的条件输入压到有界范围喂 MLP），再经 MLP 产生
      第一路预查询，随后经 FFN 产生第二路预查询。主感知第 3、6 层特征分别 RMSNorm 后沿通道拼接，经单个
      1×1 CNN 降到 planning_dim；path1 直接、path2 再过一层 FFN，分别作为两个规划 CTB 的被查询序列。8 个
      随机初始化的可学习 Mode Token 携各自预查询依次经两个 CTB，其后四层 TB 协调各 Mode。轨迹头直接回归 8 个
      扇区中线米制基线的物理残差（米），零初始化保证初始预测严格等于基线；轨迹的预测与监督全程在物理空间，
      不再经 Symlog 编解码。
"""

from __future__ import annotations

import math
from typing import Dict, Sequence

import torch
import torch.nn as nn

from config.schema import DrivingCfg
from data.target_encoding import physics_target
from model.attention import CrossAttentionBlock, RMSNormTokens, SelfAttentionBlock
from model.trajectory_decoder.checks.trajectory_decoder_checks import check_trajectory_inputs


__all__ = ["TrajectoryDecoder"]


class TrajectoryDecoder(nn.Module):
    """目标/速度条件化的 8-Mode 轨迹规划解码器。"""

    def __init__(self, cfg_driving: DrivingCfg) -> None:
        super().__init__()
        tj = cfg_driving.trajectory
        self.work_dim = cfg_driving.work_dim
        self.num_modes = tj.num_modes
        self.num_waypoints = tj.num_waypoints
        self.symlog_scale = tj.symlog_scale

        self.mode_tokens = nn.Parameter(torch.empty(1, tj.num_modes, tj.planning_dim))
        nn.init.normal_(self.mode_tokens, std=tj.mode_token_init_std)

        self.condition_encoder = nn.Sequential(
            nn.Linear(4, tj.condition_mlp_hidden), nn.SiLU(),
            nn.Linear(tj.condition_mlp_hidden, tj.planning_dim))
        self.condition_ffn = nn.Sequential(
            nn.Linear(tj.planning_dim, tj.condition_mlp_hidden), nn.SiLU(),
            nn.Linear(tj.condition_mlp_hidden, tj.planning_dim))

        self.feature_norms = nn.ModuleList(
            RMSNormTokens(self.work_dim) for _ in range(tj.cross_layers))
        self.feature_reduce = nn.Conv2d(
            tj.cross_layers * self.work_dim, tj.planning_dim, kernel_size=1)
        self.feature_ffn = nn.Sequential(
            nn.Linear(tj.planning_dim, tj.feature_ffn_hidden), nn.SiLU(),
            nn.Linear(tj.feature_ffn_hidden, tj.planning_dim))
        self.planning_cross = nn.ModuleList(
            CrossAttentionBlock(tj.planning_dim, tj.num_heads, cfg_driving.attention.mlp_ratio)
            for _ in range(tj.cross_layers))
        self.transformer = nn.ModuleList(
            SelfAttentionBlock(tj.planning_dim, tj.num_heads, cfg_driving.attention.mlp_ratio)
            for _ in range(tj.self_layers))

        self.residual_head = nn.Linear(tj.planning_dim, tj.num_waypoints * 2)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        self.confidence_head = nn.Linear(tj.planning_dim, 1)
        self.behavior_head = nn.Linear(tj.planning_dim, cfg_driving.behavior.num_classes)

        baseline = _sector_baselines(
            tj.num_modes, tj.num_waypoints, cfg_driving.bev.fov_deg, tj.baseline_step_m)
        self.register_buffer("baseline", baseline)   # 米制基线，残差直接叠加于此（物理空间）

    def forward(self, perception_features: Sequence[torch.Tensor], target_point: torch.Tensor,
                ego_velocity: torch.Tensor) -> Dict[str, torch.Tensor]:
        """以主感知第 3/6 层特征、目标点和 ego 速度解码多模态轨迹。"""
        check_trajectory_inputs(
            perception_features, target_point, ego_velocity, self.work_dim, self.num_modes)
        batch_size = int(target_point.shape[0])
        condition = physics_target(
            torch.cat((target_point, ego_velocity), dim=-1), self.symlog_scale)
        prequery_one = self.condition_encoder(condition)
        prequery_two = self.condition_ffn(prequery_one)
        prequeries = (prequery_one, prequery_two)
        contexts = self._planning_contexts(perception_features)

        tokens = self.mode_tokens.expand(batch_size, -1, -1)
        for block, prequery, context in zip(self.planning_cross, prequeries, contexts):
            tokens = block(tokens + prequery[:, None], context)
        for block in self.transformer:
            tokens = block(tokens)

        residual = self.residual_head(tokens).reshape(
            batch_size, self.num_modes, self.num_waypoints, 2)
        trajectories = self.baseline[None] + residual   # 物理空间：米制基线 + 米制残差
        return {
            "trajectory_residuals": residual,
            "trajectories": trajectories,
            "confidence": self.confidence_head(tokens).squeeze(-1),
            "behavior_logits": self.behavior_head(tokens.mean(dim=1)),
        }

    def _planning_contexts(self, features: Sequence[torch.Tensor]):
        """两层特征分别 RMSNorm、沿通道拼接后经单个 1×1 CNN 降维；path1 直接、path2 过 FFN。"""
        batch_size, _, height, width = features[0].shape
        normalized = [
            norm(feature.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(
                batch_size, self.work_dim, height, width)
            for feature, norm in zip(features, self.feature_norms)]
        reduced = self.feature_reduce(torch.cat(normalized, dim=1)).flatten(2).transpose(1, 2)
        return reduced, reduced + self.feature_ffn(reduced)


def _sector_baselines(num_modes: int, num_waypoints: int, fov_deg: float,
                      step_m: float) -> torch.Tensor:
    """沿等分视场的 8 个扇区中线生成固定米制基线 `[M,T,2]`。"""
    half_fov = math.radians(fov_deg) * 0.5
    edges = torch.linspace(-half_fov, half_fov, num_modes + 1)
    angles = 0.5 * (edges[:-1] + edges[1:])
    distances = torch.arange(1, num_waypoints + 1, dtype=torch.float32) * step_m
    directions = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
    return directions[:, None] * distances[None, :, None]
