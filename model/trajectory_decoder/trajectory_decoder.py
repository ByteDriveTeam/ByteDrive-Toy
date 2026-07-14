"""轨迹/行为联合解码器：8 扇区轨迹 Token 与行为 Token 组成同一序列，输出轨迹、置信度与多标签行为。

模块: model/trajectory_decoder/trajectory_decoder.py
依赖: torch, config.schema.DrivingCfg, model.attention.(CrossAttentionBlock, SelfAttentionBlock),
      model.trajectory_decoder.checks.trajectory_decoder_checks
读取配置:
    model.driving.work_dim
    model.driving.bev.x_min_m / x_max_m / y_min_m / y_max_m / height / width / fov_deg
    model.driving.attention.mlp_ratio
    model.driving.trajectory.num_modes / num_waypoints / token_mlp_hidden / cross_layers / self_layers /
        num_heads / velocity_norm_mps / waypoint_scale_m
    model.driving.behavior.num_classes
对外接口:
    - TrajectoryDecoder(cfg_driving) -> nn.Module
        forward(bev_feat, ego_velocity) -> dict
            # trajectories [B,M,T_wp,2] / confidence [B,M] / behavior_logits [B,C_behavior]
说明: 前向视场按 fov 均分为 num_modes 个扇区（每扇区对应一条候选轨迹）。每个扇区 Token 的初始查询由该扇区
      内 BEV cell 的均值池化特征（关键点分布代理）拼上扇区中心朝向(sin,cos)，归一化后经 Linear→SiLU→Linear
      得到。一个可学习行为 Token 追加到 M 个轨迹 Token 后，组成统一查询序列；被查询方为 BEV 特征展平的 Token
      序列，并额外并入 1 个自车速度 Token（ego 系 vx,vy 归一后线性编码）。级联 cross_layers 层 Pre-Norm
      交叉注意力聚合 BEV 与速度信息，再过 self_layers 层 Pre-Norm 自注意力使轨迹/行为互相协调。轨迹 Token
      解码航点与置信度；行为 Token 输出固定顺序的多标签 logits。精度由外层 autocast 控制。
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from config.schema import DrivingCfg
from model.attention import CrossAttentionBlock, SelfAttentionBlock
from model.trajectory_decoder.checks.trajectory_decoder_checks import check_trajectory_inputs


__all__ = ["TrajectoryDecoder"]


class TrajectoryDecoder(nn.Module):
    """8 扇区轨迹 Token + 1 个行为 Token 的联合解码器。

    Args:
        cfg_driving: 驾驶配置 `config.model.driving`。

    Shape:
        bev_feat: `[B, work_dim, Hb, Wb]`，ego_velocity: `[B, 2]`（ego 系 vx,vy）；
        输出: dict，trajectories `[B, num_modes, num_waypoints, 2]`、confidence `[B, num_modes]`、
        behavior_logits `[B, num_behavior_classes]`。
    """

    def __init__(self, cfg_driving: DrivingCfg) -> None:
        super().__init__()
        d = cfg_driving.work_dim
        tj = cfg_driving.trajectory
        self.work_dim = d
        self.num_modes = tj.num_modes
        self.num_waypoints = tj.num_waypoints
        self.num_behavior_classes = cfg_driving.behavior.num_classes
        self.velocity_norm = tj.velocity_norm_mps
        self.waypoint_scale = tj.waypoint_scale_m

        # 扇区角掩码 [M,Hb,Wb] 与扇区中心朝向 [M,2]（由 BEV 几何 + fov 推导，随模型搬设备）
        masks, dirs = _build_sectors(cfg_driving.bev, tj.num_modes)
        self.register_buffer("sector_masks", masks)
        self.register_buffer("sector_dirs", dirs)

        # 扇区 Token 初始查询：归一(池化特征 ⊕ 朝向) → Linear→SiLU→Linear
        self.token_norm = nn.LayerNorm(d + 2)
        self.token_mlp = nn.Sequential(
            nn.Linear(d + 2, tj.token_mlp_hidden), nn.SiLU(),
            nn.Linear(tj.token_mlp_hidden, d))
        # 可学习 Token 使用小幅随机初始化，避免所有样本从完全相同的零查询开始。
        self.behavior_token = nn.Parameter(torch.empty(1, 1, d))
        nn.init.normal_(self.behavior_token, mean=0.0, std=0.02)
        # 自车速度 KV Token 编码
        self.velocity_proj = nn.Linear(2, d)

        self.cross = nn.ModuleList(
            CrossAttentionBlock(d, tj.num_heads, cfg_driving.attention.mlp_ratio)
            for _ in range(tj.cross_layers))
        self.self_attn = nn.ModuleList(
            SelfAttentionBlock(d, tj.num_heads, cfg_driving.attention.mlp_ratio)
            for _ in range(tj.self_layers))

        self.waypoint_head = nn.Linear(d, tj.num_waypoints * 2)
        self.confidence_head = nn.Linear(d, 1)
        self.behavior_head = nn.Linear(d, self.num_behavior_classes)

    def forward(self, bev_feat: torch.Tensor, ego_velocity: torch.Tensor) -> Dict[str, torch.Tensor]:
        """联合解码多模态轨迹、置信度与多标签行为 logits。"""
        check_trajectory_inputs(bev_feat, ego_velocity, self.work_dim)
        b = bev_feat.shape[0]

        trajectory_tokens = self._sector_tokens(bev_feat)            # [B, M, D]
        behavior_token = self.behavior_token.expand(b, -1, -1)       # [B, 1, D]
        tokens = torch.cat((trajectory_tokens, behavior_token), dim=1)  # [B, M+1, D]
        # KV = BEV 特征 Token ⊕ 速度 Token
        bev_tokens = bev_feat.flatten(2).transpose(1, 2)             # [B, Hb*Wb, D]
        vel_token = self.velocity_proj(ego_velocity / self.velocity_norm).unsqueeze(1)  # [B,1,D]
        context = torch.cat((bev_tokens, vel_token), dim=1)

        for layer in self.cross:
            tokens = layer(tokens, context)
        for layer in self.self_attn:
            tokens = layer(tokens)

        # 乘固定尺度使 Linear 原始输出（初值 ~N(0,1)）落到米制量级，加速收敛；轨迹恒在物理空间（米）
        trajectory_tokens, behavior_token = tokens[:, :self.num_modes], tokens[:, self.num_modes]
        trajectories = self.waypoint_head(trajectory_tokens).reshape(
            b, self.num_modes, self.num_waypoints, 2) * self.waypoint_scale
        confidence = self.confidence_head(trajectory_tokens).squeeze(-1)  # [B, M]
        behavior_logits = self.behavior_head(behavior_token)         # [B, C_behavior]
        return {"trajectories": trajectories, "confidence": confidence,
                "behavior_logits": behavior_logits}

    def _sector_tokens(self, bev_feat: torch.Tensor) -> torch.Tensor:
        """每扇区均值池化 BEV 特征 ⊕ 扇区朝向 → 归一 → MLP，得初始查询 Token `[B, M, D]`。"""
        masks = self.sector_masks.to(bev_feat.dtype)                 # [M,Hb,Wb]
        counts = masks.flatten(1).sum(-1).clamp_min(1.0)            # [M]
        # [B,D,Hb,Wb] × [M,Hb,Wb] → [B,M,D] 均值池化
        pooled = torch.einsum("bdhw,mhw->bmd", bev_feat, masks) / counts[None, :, None]
        dirs = self.sector_dirs.to(bev_feat.dtype)[None].expand(bev_feat.shape[0], -1, -1)  # [B,M,2]
        token_in = self.token_norm(torch.cat((pooled, dirs), dim=-1))
        return self.token_mlp(token_in)


def _build_sectors(bev, num_modes: int):
    """前向视场按 fov 均分 num_modes 扇区：返回角掩码 [M,Hb,Wb] 与扇区中心朝向 [M,2]=(sin,cos)。"""
    import math

    x_cell = (bev.x_max_m - bev.x_min_m) / float(bev.height)
    y_cell = (bev.y_max_m - bev.y_min_m) / float(bev.width)
    # 行约定与 target_point_embedding / bev_cell_centers 一致：行 0 = 远、末行 = 近（自车在下沿）
    xs = bev.x_max_m - (torch.arange(bev.height, dtype=torch.float32) + 0.5) * x_cell
    ys = bev.y_min_m + (torch.arange(bev.width, dtype=torch.float32) + 0.5) * y_cell
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")                  # [Hb,Wb]
    # 朝向角：以前向 x 轴为 0，向右(+y)为正；仅前向 x>0 参与
    angle = torch.atan2(gy, gx.clamp_min(1e-3))                     # [Hb,Wb]
    half = math.radians(bev.fov_deg) * 0.5
    edges = torch.linspace(-half, half, num_modes + 1)             # M+1 个扇区边界
    in_fov = (gx > 0) & (angle >= -half) & (angle <= half)
    masks = torch.stack([
        (in_fov & (angle >= edges[k]) & (angle < edges[k + 1] if k < num_modes - 1
                                         else angle <= edges[k + 1])).float()
        for k in range(num_modes)], dim=0)                          # [M,Hb,Wb]
    centers = 0.5 * (edges[:-1] + edges[1:])                        # [M]
    dirs = torch.stack((torch.sin(centers), torch.cos(centers)), dim=-1)  # [M,2]
    return masks, dirs
