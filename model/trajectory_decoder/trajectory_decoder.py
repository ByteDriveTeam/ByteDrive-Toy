"""逐点数据空间流匹配轨迹解码器，使用六层 Pre-Norm CA→SA→FFN。

模块: model/trajectory_decoder/trajectory_decoder.py
依赖: torch, model.attention, model.driving_transformer.driving_transformer
读取配置: model.driving.work_dim/bev/trajectory/bev_encoder
对外接口:
    - TrajectoryDecoder(cfg_driving) -> nn.Module
说明: 训练在标准高斯与归一化 GT 之间采样直线路径；推理对速度场做 Euler 积分。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from model.attention import RMSNormTokens
from model.driving_transformer.driving_transformer import _Attention
from model.swiglu import SwiGLU
from model.trajectory_decoder.checks.trajectory_decoder_checks import check_trajectory_inputs


__all__ = ["TrajectoryDecoder"]


def _time_embedding(time, dim):
    """连续时间正余弦嵌入，频率对数均匀分布。"""
    half = dim // 2
    frequencies = torch.exp(-math.log(10000.0) * torch.arange(
        half, device=time.device, dtype=torch.float32) / half)
    phase = time[:, None] * frequencies[None]
    return torch.cat((phase.sin(), phase.cos()), -1)


class _FlowLayer(nn.Module):
    """AdaLNZero 调制三个 Pre-Norm 残差分支。"""

    def __init__(self, dim, heads, theta):
        super().__init__()
        self.ca_norm = RMSNormTokens(dim)
        self.kv_norm = RMSNormTokens(dim)
        self.sa_norm = RMSNormTokens(dim)
        self.ffn_norm = RMSNormTokens(dim)
        self.ca = _Attention(dim, heads, theta)
        self.sa = _Attention(dim, heads, theta)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), SwiGLU(dim=-1),
                                 nn.Linear(2 * dim, dim))
        self.modulation = nn.Linear(dim, 9 * dim)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, tokens, context, positions, context_positions, time_embed):
        shift_ca, scale_ca, gate_ca, shift_sa, scale_sa, gate_sa, shift_ffn, scale_ffn, gate_ffn = (
            value[:, None] for value in self.modulation(time_embed).chunk(9, -1))
        q = self.ca_norm(tokens) * (1 + scale_ca) + shift_ca
        tokens = tokens + gate_ca * self.ca(q, self.kv_norm(context),
                                            positions, context_positions)
        q = self.sa_norm(tokens) * (1 + scale_sa) + shift_sa
        tokens = tokens + gate_sa * self.sa(q, q, positions, positions)
        q = self.ffn_norm(tokens) * (1 + scale_ffn) + shift_ffn
        return tokens + gate_ffn * self.ffn(q)


class TrajectoryDecoder(nn.Module):
    """未来六秒、2Hz 的十二个并行点查询。"""

    def __init__(self, cfg_driving) -> None:
        super().__init__()
        tj, bev = cfg_driving.trajectory, cfg_driving.bev
        self.work_dim = cfg_driving.work_dim
        dim = tj.planning_dim
        self.steps = tj.num_waypoints
        self.flow_steps = tj.flow_steps
        self.noise_mode = tj.noise_mode
        self.noise_seed = tj.fixed_noise_seed
        self.symlog_scale = tj.symlog_scale
        self.register_buffer("coord_scale", torch.tensor((bev.x_max_m - bev.x_min_m,
                                                           max(abs(bev.y_min_m), abs(bev.y_max_m)))))
        self.register_buffer("future_time", torch.arange(1, self.steps + 1).float() * tj.waypoint_dt_s)
        gx = bev.x_max_m - (torch.arange(bev.height) + .5) * (
            bev.x_max_m - bev.x_min_m) / bev.height
        gy = bev.y_min_m + (torch.arange(bev.width) + .5) * (
            bev.y_max_m - bev.y_min_m) / bev.width
        xx, yy = torch.meshgrid(gx, gy, indexing="ij")
        self.register_buffer("bev_positions", torch.stack((xx, yy, torch.zeros_like(xx)),
                                                          -1).reshape(-1, 3))
        self.point_tokens = nn.Parameter(torch.randn(1, self.steps, dim) * tj.mode_token_init_std)
        self.input_projection = nn.Linear(2, dim)
        self.condition = nn.Sequential(nn.Linear(4, tj.condition_mlp_hidden), nn.SiLU(),
                                       nn.Linear(tj.condition_mlp_hidden, dim))
        self.time_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.context_norms = nn.ModuleList(RMSNormTokens(cfg_driving.work_dim)
                                           for _ in range(6))
        self.context_projections = nn.ModuleList(nn.Linear(cfg_driving.work_dim, dim)
                                                  for _ in range(6))
        self.layers = nn.ModuleList(_FlowLayer(dim, tj.num_heads, cfg_driving.bev_encoder.rope_theta)
                                    for _ in range(6))
        self.output_norm = RMSNormTokens(dim)
        self.velocity_head = nn.Linear(dim, 2)
        nn.init.zeros_(self.velocity_head.weight)
        nn.init.zeros_(self.velocity_head.bias)

    def _velocity(self, x, time, contexts, condition):
        batch = x.shape[0]
        physical = x * self.coord_scale
        positions = torch.cat((physical, self.future_time[None, :, None].expand(
            batch, -1, -1)), -1)
        tokens = self.point_tokens.expand(batch, -1, -1) + self.input_projection(x) + condition[:, None]
        time_embed = self.time_mlp(_time_embedding(time, tokens.shape[-1]))
        for layer, context in zip(self.layers, contexts):
            tokens = layer(tokens, context, positions, self.bev_positions, time_embed)
        return self.velocity_head(self.output_norm(tokens))

    def _noise(self, shape, device):
        if self.noise_mode == "random":
            return torch.randn(shape, device=device)
        generator = torch.Generator(device=device).manual_seed(self.noise_seed)
        return torch.randn(shape, device=device, generator=generator)

    def forward(self, perception_features, target_point, ego_velocity,
                trajectory=None, traj_valid=None, flow_time=None, flow_noise=None):
        """训练返回流速度目标；评估从固定或随机高斯积分到轨迹。"""
        check_trajectory_inputs(perception_features, target_point, ego_velocity,
                                self.work_dim, trajectory)
        batch = target_point.shape[0]
        condition_raw = torch.cat((target_point, ego_velocity), -1)
        condition_raw = torch.sign(condition_raw) * torch.log1p(condition_raw.abs()) * self.symlog_scale
        condition = self.condition(condition_raw)
        contexts = tuple(proj(norm(feature.flatten(2).transpose(1, 2)))
                         for feature, norm, proj in zip(
                             perception_features, self.context_norms, self.context_projections))
        if trajectory is not None:
            x0 = torch.randn_like(trajectory) if flow_noise is None else flow_noise
            x1 = trajectory / self.coord_scale
            if traj_valid is not None:
                x1 = torch.where(traj_valid.bool()[..., None], x1, x0)
            time = torch.rand(batch, device=x1.device) if flow_time is None else flow_time
            xt = (1 - time[:, None, None]) * x0 + time[:, None, None] * x1
            velocity = self._velocity(xt, time, contexts, condition)
            return {"flow_velocity": velocity, "flow_target": x1 - x0,
                    "flow_valid": traj_valid if traj_valid is not None else torch.ones_like(x1[..., 0])}
        x = self._noise((batch, self.steps, 2), target_point.device)
        for step in range(self.flow_steps):
            time = torch.full((batch,), step / self.flow_steps, device=x.device)
            x = x + self._velocity(x, time, contexts, condition) / self.flow_steps
        return {"trajectories": x * self.coord_scale}
