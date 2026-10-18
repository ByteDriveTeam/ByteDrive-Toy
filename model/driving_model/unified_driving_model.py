"""四帧单目时空 Token DrivingModel 与双流匹配解码器。

模块: model/driving_model/unified_driving_model.py
依赖: torch, config.schema.Config, model.dinov3_backbone.DinoV3Backbone
读取配置: model.dinov3_backbone.*, model.unified_driving.*, train.precision
对外接口:
    - UnifiedDrivingModel(cfg) -> nn.Module
    - forward(rgb, intrinsics, navigation_command, ego_state, motion_matrices,
              time_step=None, bev_state=None, traj_state=None) -> dict
说明: 五类导航指令作为视觉 Token 和轨迹 AdaLN 条件；轨迹流匹配在标准化
      加速度/曲率空间训练。主干默认 BF16，解码器和输出头在 FP32 中运行。
"""

from __future__ import annotations

from typing import Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.schema import Config
from model.dinov3_backbone import DinoV3Backbone
from model.driving_model.checks.unified_driving_checks import check_unified_driving_inputs
from model.privileged_bev_vae import PrivilegedBEVVAE


__all__ = ["UnifiedDrivingModel", "signed_log", "inverse_signed_log",
           "integrate_unicycle_actions"]


def signed_log(x: torch.Tensor) -> torch.Tensor:
    """把几何量映射到零点连续的 Signed-Log 空间。"""
    return torch.sign(x) * torch.log1p(x.abs())


def inverse_signed_log(x: torch.Tensor) -> torch.Tensor:
    """恢复物理坐标；训练路径禁止调用，只供推理后处理。"""
    return torch.sign(x) * torch.expm1(x.abs())


@torch.no_grad()
def integrate_unicycle_actions(actions: torch.Tensor, ego_state: torch.Tensor,
                               action_mean=(0.0, 0.0), action_std=(1.0, 1.0),
                               dt_s: float = 0.1) -> torch.Tensor:
    """推理阶段反标准化并积分 [a, kappa]，返回 [x, y, theta] 航点。"""
    if actions.ndim != 3 or actions.shape[-1] != 2:
        raise ValueError("actions 必须为 [B,30,2]")
    if ego_state.ndim != 2 or ego_state.shape[0] != actions.shape[0] or ego_state.shape[-1] < 2:
        raise ValueError("ego_state 必须为 [B,4]，且前两维为速度")
    mean = actions.new_tensor(action_mean).view(1, 1, 2)
    std = actions.new_tensor(action_std).view(1, 1, 2)
    physical = actions * std + mean
    velocity = inverse_signed_log(ego_state[..., :2].float()).norm(dim=-1)
    position = actions.new_zeros(actions.shape[0], 2)
    heading = actions.new_zeros(actions.shape[0])
    waypoints = []
    for control in physical.unbind(dim=1):
        acceleration, curvature = control.unbind(dim=-1)
        position = position + velocity.unsqueeze(-1) * torch.stack((heading.cos(), heading.sin()), dim=-1) * dt_s
        heading = heading + velocity * curvature * dt_s
        velocity = (velocity + acceleration * dt_s).clamp_min(0.0)
        waypoints.append(torch.cat((position, heading.unsqueeze(-1)), dim=-1))
    return torch.stack(waypoints, dim=1)


class _SwiGLU(nn.Module):
    """384 维 Token 的 SwiGLU 前馈层。"""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(dim, hidden * 2)
        self.out_proj = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        return self.out_proj(value * F.silu(gate))


def _qk_norm(x: torch.Tensor) -> torch.Tensor:
    """对每个注意力头的 Q/K 做 FP32 LayerNorm。"""
    return F.layer_norm(x.float(), (x.shape[-1],)).to(x.dtype)


def _rotate(features: torch.Tensor, positions: torch.Tensor, axis_dims, theta: float):
    """按坐标轴切分特征并应用旋转位置编码。"""
    result, cursor = [], 0
    features, positions = features.float(), positions.float()
    for axis, width in enumerate(axis_dims):
        part = features[..., cursor:cursor + width]
        pairs = width // 2
        index = torch.arange(pairs, device=features.device, dtype=torch.float32)
        inverse = torch.pow(features.new_tensor(theta), -2.0 * index / width)
        angles = positions[..., axis, None] * inverse
        even, odd = part[..., 0::2], part[..., 1::2]
        cos, sin = angles.cos(), angles.sin()
        result.append(torch.stack((even * cos - odd * sin,
                                   even * sin + odd * cos), dim=-1).flatten(-2))
        cursor += width
    if cursor < features.shape[-1]:
        result.append(features[..., cursor:])
    return torch.cat(result, dim=-1)


class _Attention(nn.Module):
    """带 QK-Norm、SDPA 和可选 RoPE 的多头注意力。"""

    def __init__(self, dim: int, heads: int, rope_axes=None, rope_theta: float = 100.0,
                 use_geometry: bool = False):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("注意力通道数必须能被头数整除")
        self.heads = heads
        self.head_dim = dim // heads
        self.rope_axes = tuple(rope_axes) if rope_axes else None
        self.rope_theta = float(rope_theta)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.geo_q = nn.Linear(dim, dim, bias=False) if use_geometry else None
        self.geo_k = nn.Linear(dim, dim, bias=False) if use_geometry else None

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        return x.reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, query, context=None, positions=None, q_geometry=None, k_geometry=None):
        context = query if context is None else context
        q_input = self.q(query)
        k_input = self.k(context)
        if self.geo_q is not None and q_geometry is not None:
            q_input = q_input + self.geo_q(q_geometry)
        if self.geo_k is not None and k_geometry is not None:
            k_input = k_input + self.geo_k(k_geometry)
        q, k, v = self._split(q_input), self._split(k_input), self._split(self.v(context))
        q, k = _qk_norm(q), _qk_norm(k)
        if self.rope_axes is not None and positions is not None:
            rope_positions = positions.unsqueeze(1) if positions.ndim == 3 else positions
            q = _rotate(q, rope_positions, self.rope_axes, self.rope_theta)
            k = _rotate(k, rope_positions, self.rope_axes, self.rope_theta)
        output = F.scaled_dot_product_attention(q, k, v)
        return self.o(output.transpose(1, 2).reshape(query.shape[0], query.shape[1], -1))


class _TransformerBlock(nn.Module):
    """视觉主干的标准 Pre-Norm Transformer 大块。"""

    def __init__(self, dim: int, heads: int, hidden: int, theta: float):
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = _Attention(dim, heads, (16, 16, 16), theta)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = _SwiGLU(dim, hidden)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x), positions=positions)
        return x + self.ffn(self.norm_ffn(x))


class _AdaLNZeroBlock(nn.Module):
    """带 AdaLN-Zero 的 CA→SA→FFN 残差大块。"""

    def __init__(self, dim: int, cross_heads: int, self_heads: int,
                 hidden: int, rope_axes, theta: float):
        super().__init__()
        self.norm_ca, self.norm_context = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ca = _Attention(dim, cross_heads, None, theta, use_geometry=True)
        self.norm_sa, self.sa = nn.LayerNorm(dim), _Attention(dim, self_heads, rope_axes, theta)
        self.norm_ffn, self.ffn = nn.LayerNorm(dim), _SwiGLU(dim, hidden)
        self.condition = nn.Linear(dim, dim * 9)
        nn.init.zeros_(self.condition.weight)
        nn.init.zeros_(self.condition.bias)

    @staticmethod
    def _modulate(x, shift, scale):
        return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x, context, condition, positions, q_geometry=None, k_geometry=None):
        params = self.condition(condition).chunk(9, dim=-1)
        ca_shift, ca_scale, ca_gate = params[:3]
        sa_shift, sa_scale, sa_gate = params[3:6]
        ff_shift, ff_scale, ff_gate = params[6:9]
        x = x + ca_gate.unsqueeze(1) * self.ca(
            self._modulate(self.norm_ca(x), ca_shift, ca_scale), self.norm_context(context),
            q_geometry=q_geometry, k_geometry=k_geometry)
        x = x + sa_gate.unsqueeze(1) * self.sa(
            self._modulate(self.norm_sa(x), sa_shift, sa_scale), positions=positions)
        return x + ff_gate.unsqueeze(1) * self.ffn(
            self._modulate(self.norm_ffn(x), ff_shift, ff_scale))


class _ConditionMLP(nn.Module):
    """把几何或状态条件投影到模型宽度。"""

    def __init__(self, input_dim: int, dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.SiLU(), nn.Linear(hidden, dim))

    def forward(self, x):
        return self.net(x)


def _sinusoidal_time(time_step: torch.Tensor, dim: int) -> torch.Tensor:
    """用多频正余弦生成时间条件。"""
    half = dim // 2
    index = torch.arange(half, device=time_step.device, dtype=torch.float32)
    frequencies = torch.exp(-torch.log(time_step.new_tensor(10000.0)) * index / half)
    angles = time_step.float().unsqueeze(-1) * frequencies
    return torch.cat((angles.sin(), angles.cos()), dim=-1)


class _VisualProcessor(nn.Module):
    """四帧 DINO 融合、3D RoPE 主干和 Patch 网格恢复。"""

    def __init__(self, cfg: Config):
        super().__init__()
        bb, spec = cfg.model.dinov3_backbone, cfg.model.unified_driving
        self.dim, self.patch_size = spec.dim, bb.patch_size
        self.num_register_tokens = spec.num_register_tokens
        self.num_annotation_queries = spec.annotation_query_count
        self.special_tokens = 2 + self.num_register_tokens + self.num_annotation_queries
        self.register_start = 2
        self.annotation_start = self.register_start + self.num_register_tokens
        self.patch_start = self.special_tokens
        self.backbone = DinoV3Backbone(bb)
        self.dino_feature_norms = nn.ModuleList(
            [nn.LayerNorm(bb.hidden_dim) for _ in bb.feature_layers])
        self.fusion = nn.Linear(bb.hidden_dim * len(bb.feature_layers), spec.dim)
        self.downsample = nn.Conv2d(spec.dim, spec.dim, 2, stride=2)
        self.navigation_token = nn.Embedding(len(spec.navigation_commands), spec.dim)
        self.motion_token = nn.Linear(4, spec.dim)
        self.register_tokens = nn.Parameter(torch.randn(self.num_register_tokens, spec.dim) * 0.02)
        self.annotation_queries = nn.Parameter(
            torch.randn(self.num_annotation_queries, spec.dim) * 0.02)
        self.blocks = nn.ModuleList([_TransformerBlock(
            spec.dim, spec.self_heads_3d, spec.ffn_hidden, spec.rope_theta) for _ in range(8)])
        self.aggregate = nn.Sequential(nn.LayerNorm(spec.dim * 4), nn.Linear(spec.dim * 4, spec.dim))
        self.restore_expand = nn.Conv2d(spec.dim, spec.dim * 4, 3, padding=1, bias=True)
        nn.init.zeros_(self.restore_expand.bias)
        self.restore_shuffle = nn.PixelShuffle(2)
        self.restore_conv = nn.Conv2d(spec.dim, spec.dim, 3, padding=1)
        self.restore_residual = nn.Sequential(
            nn.Conv2d(spec.dim, spec.dim, 3, padding=1), nn.SiLU(),
            nn.Conv2d(spec.dim, spec.dim, 3, padding=1))
        self.restore_norm = nn.LayerNorm(spec.dim)
        self.token_output_norm = nn.LayerNorm(spec.dim)
        self.geometry_output_norm = nn.LayerNorm(spec.dim)
        self.image_geometry = _ConditionMLP(3, spec.dim, spec.geometry_hidden)
        self.annotation_state_head = nn.Linear(spec.dim, spec.annotation_state_classes)
        self.annotation_reason_head = nn.Linear(spec.dim, spec.annotation_reason_classes)
        self.annotation_probability_head = nn.Linear(spec.dim, spec.annotation_probability_dim)

    def forward(self, rgb, navigation_command, ego_state, intrinsics, motion_matrices):
        batch, frames, channels, height, width = rgb.shape
        with torch.no_grad():
            selected = self.backbone(rgb.reshape(batch * frames, channels, height, width))
        patch_count = (height // self.patch_size) * (width // self.patch_size)
        layer_patches = selected[:, :, -patch_count:, :]
        # 四个 DINO 层分别标准化，再沿特征维拼接，避免不同深度的尺度差异进入融合层。
        layer_patches = torch.stack(
            [norm(layer_patches[:, index].float())
             for index, norm in enumerate(self.dino_feature_norms)], dim=1)
        patches = layer_patches.permute(0, 2, 1, 3).reshape(
            batch * frames, patch_count, -1)
        fused = self.fusion(patches).transpose(1, 2).reshape(
            batch * frames, self.dim, height // self.patch_size, width // self.patch_size)
        fused = self.downsample(fused)
        small_h, small_w = fused.shape[-2:]
        patches = fused.flatten(2).transpose(1, 2).reshape(batch, frames * small_h * small_w, self.dim)
        motion_token = self.motion_token(ego_state)
        condition = torch.stack((self.navigation_token(navigation_command).to(motion_token.dtype),
                                 motion_token), dim=1)
        registers = self.register_tokens.unsqueeze(0).expand(batch, -1, -1)
        annotation_queries = self.annotation_queries.unsqueeze(0).expand(batch, -1, -1)
        tokens = torch.cat((condition, registers, annotation_queries, patches), dim=1)
        positions = self._positions(batch, frames, small_h, small_w, tokens.device, self.special_tokens)
        outputs = []
        for index, block in enumerate(self.blocks, 1):
            tokens = block(tokens, positions)
            if index in (2, 4, 6, 8):
                outputs.append(tokens[:, self.patch_start:])
        # 聚合、PixelShuffle、残差卷积和末端 LayerNorm 固定在 FP32。
        with torch.autocast(device_type=outputs[-1].device.type, enabled=False):
            normalized = [F.layer_norm(item.float(), (self.dim,)) for item in outputs]
            aggregated = self.aggregate(torch.cat(normalized, dim=-1))
            aggregated = aggregated.reshape(batch * frames, small_h, small_w, self.dim).permute(0, 3, 1, 2)
            restored = self.restore_conv(self.restore_shuffle(self.restore_expand(aggregated)))
            restored = restored + self.restore_residual(restored)
            restored_tokens = self.restore_norm(restored.flatten(2).transpose(1, 2))
        restored = restored_tokens.transpose(1, 2).reshape(
            batch, frames, self.dim, height // self.patch_size, width // self.patch_size)
        tokens = self.token_output_norm(tokens.float()).to(restored.dtype)
        image_geometry = self._ray_geometry(
            intrinsics, motion_matrices, batch, frames, small_h, small_w, self.patch_size)
        image_geometry = self.geometry_output_norm(image_geometry.float()).to(restored.dtype)
        geometry_tokens = image_geometry.reshape(batch, frames * small_h * small_w, self.dim)
        annotation_tokens = tokens[:, self.annotation_start:self.patch_start]
        annotation_tokens = annotation_tokens.float()
        with torch.autocast(device_type=annotation_tokens.device.type, enabled=False):
            annotation = {
                "tokens": annotation_tokens,
                "state_logits": self.annotation_state_head(annotation_tokens[:, 0]),
                "reason_logits": self.annotation_reason_head(annotation_tokens[:, 1]),
                "probability_logits": self.annotation_probability_head(annotation_tokens[:, 2]),
            }
        return tokens, torch.cat((geometry_tokens.new_zeros(batch, self.special_tokens, self.dim), geometry_tokens), dim=1), restored, annotation

    @staticmethod
    def _positions(batch, frames, height, width, device, special_tokens):
        y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
        spatial = torch.stack((x, y), dim=-1).reshape(-1, 2).float()
        frame = torch.arange(frames, device=device, dtype=torch.float32).view(-1, 1, 1).expand(frames, height * width, 1)
        patch = torch.cat((spatial.unsqueeze(0).expand(frames, -1, -1), frame), dim=-1).reshape(-1, 3)
        return torch.cat((torch.zeros(special_tokens, 3, device=device), patch), dim=0).unsqueeze(0).expand(batch, -1, -1)

    def _ray_geometry(self, intrinsics, motion_matrices, batch, frames, height, width, patch_size):
        device = intrinsics.device
        yy, xx = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32) * (2.0 * patch_size) + patch_size,
            torch.arange(width, device=device, dtype=torch.float32) * (2.0 * patch_size) + patch_size,
            indexing="ij")
        yy, xx = yy.reshape(1, 1, -1, 1), xx.reshape(1, 1, -1, 1)
        fx, fy = intrinsics[..., 0:1], intrinsics[..., 1:2]
        cx, cy = intrinsics[..., 2:3], intrinsics[..., 3:4]
        depth = torch.arange(0.1, 64.01, 0.1, device=device).view(1, 1, 1, -1)
        forward = depth.expand(batch, frames, height * width, -1)
        lateral = forward * (xx - cx.unsqueeze(-1)) / fx.unsqueeze(-1)
        vertical = forward * (yy - cy.unsqueeze(-1)) / fy.unsqueeze(-1)
        points = torch.stack((forward, lateral, vertical), dim=-1)
        if motion_matrices.shape[-1] == 4:
            homogeneous = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)
            points = torch.einsum("btij,btnsj->btnsi", motion_matrices.float(), homogeneous)[..., :3]
        else:
            xy = torch.cat((points[..., :2], torch.ones_like(points[..., :1])), dim=-1)
            transformed = torch.einsum("btij,btnsj->btnsi", motion_matrices.float(), xy)
            points = torch.cat((transformed[..., :2], points[..., 2:3]), dim=-1)
        # 射线物理坐标只做线性尺度归一化；不把位置编码直接加到视觉特征。
        points = (points / points.new_tensor([64.0, 64.0, 64.0])).mean(dim=3)
        return self.image_geometry(points)


class _BEVDecoder(nn.Module):
    """六帧通道拼接的 BEV latent 流匹配解码器。"""

    def __init__(self, spec, geometry):
        super().__init__()
        self.dim, self.latent_dim = spec.dim, spec.bev_latent_dim
        self.input = nn.Conv2d(self.latent_dim * 6, self.dim, 1)
        self.blocks = nn.ModuleList([_AdaLNZeroBlock(
            self.dim, spec.cross_heads, spec.self_heads_2d, spec.ffn_hidden,
            (32, 32), spec.rope_theta) for _ in range(2)])
        self.output_norm, self.output = nn.LayerNorm(self.dim), nn.Conv2d(self.dim, self.latent_dim * 6, 1)
        self.geometry = geometry
        self.geometry_norm = nn.LayerNorm(self.dim)

    def forward(self, state, visual_tokens, visual_geometry, time_step):
        batch, _, height, width = state.shape
        query = self.input(state).flatten(2).transpose(1, 2)
        y, x = torch.meshgrid(torch.arange(height, device=state.device), torch.arange(width, device=state.device), indexing="ij")
        position = torch.stack((x, y), dim=-1).reshape(1, -1, 2).float().expand(batch, -1, -1)
        grid_x = x.float() / max(width - 1, 1)
        grid_y = y.float() / max(height - 1, 1) * 2.0 - 1.0
        centers = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1), torch.zeros_like(grid_x).reshape(-1)), dim=-1)
        z = torch.arange(-3.0, 16.01, 0.1, device=state.device).view(1, 1, -1, 1)
        z = (z + 3.0) / 19.0 * 2.0 - 1.0
        bev_xy = centers[:, :2].view(1, -1, 1, 2).expand(batch, -1, z.shape[2], -1)
        bev_points = torch.cat((bev_xy, z.expand(batch, bev_xy.shape[1], -1, 1)), dim=-1)
        geometry = self.geometry_norm(self.geometry(bev_points.mean(dim=2)))
        condition = _sinusoidal_time(time_step, self.dim)
        for block in self.blocks:
            query = block(query, visual_tokens, condition, position, geometry, visual_geometry)
        return self.output(self.output_norm(query).transpose(1, 2).reshape(batch, self.dim, height, width))


class _TrajectoryDecoder(nn.Module):
    """三维 RoPE 的30点单轨迹流匹配解码器。"""

    def __init__(self, spec, geometry):
        super().__init__()
        self.dim, self.points = spec.dim, spec.num_waypoints
        self.state = nn.Linear(2, self.dim)
        self.navigation = nn.Embedding(len(spec.navigation_commands), self.dim)
        self.ego = _ConditionMLP(4, self.dim, spec.condition_hidden)
        self.time = _ConditionMLP(self.dim, self.dim, spec.condition_hidden)
        self.num_annotation_queries = spec.annotation_query_count
        self.annotation_queries = nn.Parameter(
            torch.randn(self.num_annotation_queries, self.dim) * 0.02)
        self.blocks = nn.ModuleList([_AdaLNZeroBlock(
            self.dim, spec.cross_heads, spec.self_heads_3d, spec.ffn_hidden,
            (16, 16, 16), spec.rope_theta) for _ in range(6)])
        self.output_norms = nn.ModuleList([nn.LayerNorm(self.dim) for _ in range(6)])
        self.action_heads = nn.ModuleList([nn.Linear(self.dim, 2) for _ in range(6)])
        self.geometry = geometry
        self.geometry_norm = nn.LayerNorm(self.dim)
        self.annotation_state_head = nn.Linear(self.dim, spec.annotation_state_classes)
        self.annotation_reason_head = nn.Linear(self.dim, spec.annotation_reason_classes)
        self.annotation_probability_head = nn.Linear(self.dim, spec.annotation_probability_dim)

    def forward(self, state, visual_tokens, visual_geometry, time_step, navigation_command, ego_state):
        batch = state.shape[0]
        query = self.state(state)
        point_time = torch.linspace(0.1, 3.0, self.points, device=state.device).view(1, -1, 1).expand(batch, -1, -1)
        positions = torch.cat((state, point_time), dim=-1)
        z = torch.arange(-3.0, 16.01, 0.1, device=state.device).view(1, 1, -1, 1)
        z = (z + 3.0) / 19.0 * 2.0 - 1.0
        # 动作状态处于标准化空间，位置分支不对控制量做逆变换。
        trajectory_xy = state / 4.0
        trajectory_points = trajectory_xy.unsqueeze(2).expand(batch, self.points, z.shape[2], 2)
        trajectory_points = torch.cat((trajectory_points, z.expand(batch, self.points, -1, 1)), dim=-1)
        geometry = self.geometry_norm(self.geometry(trajectory_points.mean(dim=2)))
        annotation_queries = self.annotation_queries.unsqueeze(0).expand(batch, -1, -1)
        query = torch.cat((query, annotation_queries), dim=1)
        positions = torch.cat((positions, positions.new_zeros(batch, self.num_annotation_queries, 3)), dim=1)
        geometry = torch.cat((geometry, geometry.new_zeros(batch, self.num_annotation_queries, self.dim)), dim=1)
        condition = (self.time(_sinusoidal_time(time_step, self.dim))
                     + self.navigation(navigation_command) + self.ego(ego_state))
        fields = []
        for block, norm, head in zip(self.blocks, self.output_norms, self.action_heads):
            query = block(query, visual_tokens, condition, positions, geometry, visual_geometry)
            fields.append(head(norm(query[:, :self.points])))
        annotation_tokens = self.output_norms[-1](query[:, self.points:]).float()
        annotation = {
            "tokens": annotation_tokens,
            "state_logits": self.annotation_state_head(annotation_tokens[:, 0]),
            "reason_logits": self.annotation_reason_head(annotation_tokens[:, 1]),
            "probability_logits": self.annotation_probability_head(annotation_tokens[:, 2]),
        }
        return {"flows": fields, "annotation": annotation}


class UnifiedDrivingModel(nn.Module):
    """四帧前向单目 DrivingModel，输出 BEV 与轨迹速度场。"""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.visual = _VisualProcessor(cfg)
        spec = cfg.model.unified_driving
        self.shared_geometry = _ConditionMLP(3, spec.dim, spec.geometry_hidden)
        self.bev_decoder = _BEVDecoder(spec, self.shared_geometry)
        self.trajectory_decoder = _TrajectoryDecoder(spec, self.shared_geometry)
        self.bev_vae = PrivilegedBEVVAE(cfg)
        self.bev_vae.requires_grad_(False)
        self.bev_vae.eval()
        self.precision = cfg.train.precision

    @staticmethod
    def integrate_actions(actions, ego_state, action_mean=(0.0, 0.0), action_std=(1.0, 1.0),
                          dt_s=0.1):
        """推理后处理：把标准化控制量积分为连续航点。"""
        return integrate_unicycle_actions(actions, ego_state, action_mean, action_std, dt_s)

    @torch.no_grad()
    def sample_trajectory_actions(self, rgb, intrinsics, navigation_command, ego_state,
                                  motion_matrices, steps: int = 10):
        """从标准高斯噪声用 Euler 流匹配采样标准化 [加速度, 曲率]。"""
        if steps <= 0:
            raise ValueError("steps 必须为正整数")
        check_unified_driving_inputs(rgb, intrinsics, navigation_command, ego_state, motion_matrices,
                                     len(self.cfg.model.unified_driving.navigation_commands))
        enabled = self.precision == "bf16" and rgb.device.type != "cpu"
        with torch.autocast(device_type=rgb.device.type, dtype=torch.bfloat16, enabled=enabled):
            tokens, geometry, _, _ = self.visual(
                rgb, navigation_command, ego_state, intrinsics, motion_matrices)
        state = torch.randn(
            rgb.shape[0], self.trajectory_decoder.points, 2,
            device=rgb.device, dtype=torch.float32)
        with torch.autocast(device_type=rgb.device.type, dtype=torch.bfloat16, enabled=False):
            tokens, geometry = tokens.float(), geometry.float()
            for index in range(steps):
                time_step = torch.full(
                    (rgb.shape[0],), index / float(steps), device=rgb.device, dtype=torch.float32)
                flow = self.trajectory_decoder(
                    state, tokens, geometry, time_step,
                    navigation_command, ego_state.float())["flows"][-1]
                state = state + flow.float() / float(steps)
        return state

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """返回除冻结 DINOv3 外的可训练参数。"""
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def train(self, mode: bool = True):
        """切换训练状态，同时保持 DINOv3 和冻结 VAE 为 eval。"""
        super().train(mode)
        self.visual.backbone.eval()
        self.bev_vae.eval()
        return self

    @torch.no_grad()
    def encode_bev_targets(self, bev_grid: torch.Tensor) -> torch.Tensor:
        """在线编码六帧 BEV 栅格；VAE 冻结且不保存 latent 缓存。"""
        if bev_grid.ndim != 5 or bev_grid.shape[2] != self.bev_vae.in_channels:
            raise ValueError("bev_grid 必须为 [B,6,12,H,W]")
        batch, frames = bev_grid.shape[:2]
        encoded = self.bev_vae.encode(
            bev_grid.reshape(batch * frames, *bev_grid.shape[2:]).float(), sample=False)["mu"]
        return encoded.reshape(batch, frames, *encoded.shape[1:])

    def forward(self, rgb: torch.Tensor, intrinsics: torch.Tensor,
                navigation_command: torch.Tensor, ego_state: torch.Tensor,
                motion_matrices: torch.Tensor, time_step: Optional[torch.Tensor] = None,
                bev_state: Optional[torch.Tensor] = None,
                traj_state: Optional[torch.Tensor] = None):
        """处理四帧图像，并在给定流状态时输出两个速度场。"""
        check_unified_driving_inputs(rgb, intrinsics, navigation_command, ego_state, motion_matrices,
                                     len(self.cfg.model.unified_driving.navigation_commands))
        enabled = self.precision == "bf16" and rgb.device.type != "cpu"
        with torch.autocast(device_type=rgb.device.type, dtype=torch.bfloat16, enabled=enabled):
            tokens, geometry, feature_maps, visual_annotation = self.visual(
                rgb, navigation_command, ego_state, intrinsics, motion_matrices)
        outputs = {"visual_tokens": tokens, "image_geometry": geometry,
                   "visual_features": feature_maps,
                   "visual_annotation": visual_annotation}
        with torch.autocast(device_type=rgb.device.type, dtype=torch.bfloat16, enabled=False):
            tokens_fp32 = tokens.float()
            if time_step is not None and bev_state is not None:
                outputs["bev_flow"] = self.bev_decoder(
                    bev_state.float(), tokens_fp32, geometry.float(), time_step.float())
            if time_step is not None and traj_state is not None:
                trajectory_output = self.trajectory_decoder(
                    traj_state.float(), tokens_fp32, geometry.float(), time_step.float(),
                    navigation_command, ego_state.float())
                outputs["traj_flows"] = trajectory_output["flows"]
                outputs["traj_annotation"] = trajectory_output["annotation"]
        return outputs
