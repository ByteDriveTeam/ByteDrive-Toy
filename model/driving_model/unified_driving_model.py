"""四帧单目时空 Token DrivingModel 与双流匹配解码器。

模块: model/driving_model/unified_driving_model.py
依赖: torch, config.schema.Config, model.dinov3_backbone.DinoV3Backbone
读取配置: model.dinov3_backbone.*, model.unified_driving.*, train.precision
对外接口:
    - UnifiedDrivingModel(cfg) -> nn.Module
    - forward(rgb, intrinsics, target_point, ego_state, motion_matrices,
              time_step=None, bev_state=None, traj_state=None) -> dict
说明: 轨迹流匹配在 Signed-Log 数值空间训练；逆变换仅供推理结束后的脱离
      梯度后处理。主干默认 BF16，分支末端和输出头在 FP32 中运行。
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


__all__ = ["UnifiedDrivingModel", "signed_log", "inverse_signed_log"]


def signed_log(x: torch.Tensor) -> torch.Tensor:
    """把几何量映射到零点连续的 Signed-Log 空间。"""
    return torch.sign(x) * torch.log1p(x.abs())


def inverse_signed_log(x: torch.Tensor) -> torch.Tensor:
    """恢复物理坐标；训练路径禁止调用，只供推理后处理。"""
    return torch.sign(x) * torch.expm1(x.abs())


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


def _fourier_target(target_point: torch.Tensor, bands: int = 32) -> torch.Tensor:
    """对线性归一化目标点做高频正余弦编码。"""
    frequencies = torch.logspace(
        0.0, 15.0, bands, base=2.0, device=target_point.device, dtype=target_point.dtype)
    angles = target_point.unsqueeze(-1) * frequencies * target_point.new_tensor(3.141592653589793)
    encoded = torch.cat((angles.sin(), angles.cos()), dim=-1)
    return encoded.flatten(start_dim=-2)


class _VisualProcessor(nn.Module):
    """四帧 DINO 融合、3D RoPE 主干和 Patch 网格恢复。"""

    def __init__(self, cfg: Config):
        super().__init__()
        bb, spec = cfg.model.dinov3_backbone, cfg.model.unified_driving
        self.dim, self.patch_size = spec.dim, bb.patch_size
        self.num_register_tokens = spec.num_register_tokens
        self.special_tokens = 2 + self.num_register_tokens
        self.backbone = DinoV3Backbone(bb)
        self.fusion = nn.Linear(bb.hidden_dim * 4, spec.dim)
        self.downsample = nn.Conv2d(spec.dim, spec.dim, 2, stride=2)
        self.target_token, self.motion_token = nn.Linear(2, spec.dim), nn.Linear(4, spec.dim)
        self.register_tokens = nn.Parameter(torch.randn(self.num_register_tokens, spec.dim) * 0.02)
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

    def forward(self, rgb, target_point, ego_state, intrinsics, motion_matrices):
        batch, frames, channels, height, width = rgb.shape
        with torch.no_grad():
            selected = self.backbone(rgb.reshape(batch * frames, channels, height, width))
        patch_count = (height // self.patch_size) * (width // self.patch_size)
        patches = selected[:, :, -patch_count:, :].permute(0, 2, 1, 3).reshape(batch * frames, patch_count, -1)
        fused = self.fusion(patches).transpose(1, 2).reshape(
            batch * frames, self.dim, height // self.patch_size, width // self.patch_size)
        fused = self.downsample(fused)
        small_h, small_w = fused.shape[-2:]
        patches = fused.flatten(2).transpose(1, 2).reshape(batch, frames * small_h * small_w, self.dim)
        condition = torch.stack((self.target_token(target_point), self.motion_token(ego_state)), dim=1)
        registers = self.register_tokens.unsqueeze(0).expand(batch, -1, -1)
        tokens = torch.cat((condition, registers, patches), dim=1)
        positions = self._positions(batch, frames, small_h, small_w, tokens.device, self.special_tokens)
        outputs = []
        for index, block in enumerate(self.blocks, 1):
            tokens = block(tokens, positions)
            if index in (2, 4, 6, 8):
                outputs.append(tokens[:, self.special_tokens:])
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
        return tokens, torch.cat((geometry_tokens.new_zeros(batch, self.special_tokens, self.dim), geometry_tokens), dim=1), restored

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
        self.target, self.ego = _ConditionMLP(128, self.dim, spec.condition_hidden), _ConditionMLP(4, self.dim, spec.condition_hidden)
        self.time = _ConditionMLP(self.dim, self.dim, spec.condition_hidden)
        self.blocks = nn.ModuleList([_AdaLNZeroBlock(
            self.dim, spec.cross_heads, spec.self_heads_3d, spec.ffn_hidden,
            (16, 16, 16), spec.rope_theta) for _ in range(6)])
        self.output_norms = nn.ModuleList([nn.LayerNorm(self.dim) for _ in range(6)])
        self.velocity_heads = nn.ModuleList([nn.Linear(self.dim, 2) for _ in range(6)])
        self.geometry = geometry
        self.geometry_norm = nn.LayerNorm(self.dim)

    def forward(self, state, visual_tokens, visual_geometry, time_step, target_point, ego_state):
        batch = state.shape[0]
        query = self.state(state)
        point_time = torch.linspace(0.1, 3.0, self.points, device=state.device).view(1, -1, 1).expand(batch, -1, -1)
        positions = torch.cat((state, point_time), dim=-1)
        z = torch.arange(-3.0, 16.01, 0.1, device=state.device).view(1, 1, -1, 1)
        z = (z + 3.0) / 19.0 * 2.0 - 1.0
        # state 已处于 Signed-Log 训练空间，只做线性尺度归一化，不逆变换。
        trajectory_xy = state / 4.0
        trajectory_points = trajectory_xy.unsqueeze(2).expand(batch, self.points, z.shape[2], 2)
        trajectory_points = torch.cat((trajectory_points, z.expand(batch, self.points, -1, 1)), dim=-1)
        geometry = self.geometry_norm(self.geometry(trajectory_points.mean(dim=2)))
        condition = self.time(_sinusoidal_time(time_step, self.dim)) + self.target(_fourier_target(target_point)) + self.ego(ego_state)
        fields = []
        for block, norm, head in zip(self.blocks, self.output_norms, self.velocity_heads):
            query = block(query, visual_tokens, condition, positions, geometry, visual_geometry)
            fields.append(head(norm(query)))
        return fields


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
                target_point: torch.Tensor, ego_state: torch.Tensor,
                motion_matrices: torch.Tensor, time_step: Optional[torch.Tensor] = None,
                bev_state: Optional[torch.Tensor] = None,
                traj_state: Optional[torch.Tensor] = None):
        """处理四帧图像，并在给定流状态时输出两个速度场。"""
        check_unified_driving_inputs(rgb, intrinsics, target_point, ego_state, motion_matrices)
        enabled = self.precision == "bf16" and rgb.device.type != "cpu"
        with torch.autocast(device_type=rgb.device.type, dtype=torch.bfloat16, enabled=enabled):
            tokens, geometry, feature_maps = self.visual(
                rgb, target_point, ego_state, intrinsics, motion_matrices)
        outputs = {"visual_tokens": tokens, "image_geometry": geometry,
                   "visual_features": feature_maps}
        with torch.autocast(device_type=rgb.device.type, dtype=torch.bfloat16, enabled=False):
            tokens_fp32 = tokens.float()
            if time_step is not None and bev_state is not None:
                outputs["bev_flow"] = self.bev_decoder(
                    bev_state.float(), tokens_fp32, geometry.float(), time_step.float())
            if time_step is not None and traj_state is not None:
                outputs["traj_flows"] = self.trajectory_decoder(
                    traj_state.float(), tokens_fp32, geometry.float(), time_step.float(),
                    target_point.float(), ego_state.float())
        return outputs
