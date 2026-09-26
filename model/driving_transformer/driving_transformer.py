"""六层 Pre-Norm CA→SA→FFN 驾驶感知主干，位置只提示注意力。

模块: model/driving_transformer/driving_transformer.py
依赖: torch, model.rope_3d, model.attention, model.swiglu, model.lidar_fusion
读取配置: model.driving.work_dim/attention/bev/bev_encoder/frustum/query/detection/lidar_fusion
对外接口:
    - DrivingTransformer(cfg) -> nn.Module
说明: 图像射线与 BEV/Detect 几何由同构同初值的两个 MLP 编码，只加入 Q/K；
      V 和残差 Token 始终保持纯内容。每层完成图像 CA 后可在 1/3/5 层融合 LiDAR。
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention import RMSNormTokens
from model.driving_transformer.checks.driving_transformer_checks import check_transformer_inputs
from model.rope_3d import apply_rope_3d
from model.swiglu import SwiGLU


__all__ = ["DrivingTransformer"]


def _symlog(x: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs()) * scale


class _Attention(nn.Module):
    """独立 Q/K/V 投影；几何只影响 Q/K，RoPE 不作用于 V。"""

    def __init__(self, dim: int, heads: int, theta: float) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.theta = theta
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        # 每头分给 XY 和时间三个偶数轴，其余维度保持不旋转。
        rotary = (self.head_dim // 6) * 2
        self.axis_dims = (rotary, rotary, rotary)

    def forward(self, query, context, q_pos=None, k_pos=None,
                q_hint=None, k_hint=None, key_valid=None):
        b, nq, dim = query.shape
        nk = context.shape[1]
        q = self.q(query + q_hint if q_hint is not None else query)
        k = self.k(context + k_hint if k_hint is not None else context)
        v = self.v(context)
        q = q.reshape(b, nq, self.heads, self.head_dim).transpose(1, 2)
        k = k.reshape(b, nk, self.heads, self.head_dim).transpose(1, 2)
        v = v.reshape(b, nk, self.heads, self.head_dim).transpose(1, 2)
        if q_pos is not None:
            q = apply_rope_3d(q, q_pos, self.axis_dims, self.theta).to(v.dtype)
        if k_pos is not None:
            k = apply_rope_3d(k, k_pos, self.axis_dims, self.theta).to(v.dtype)
        mask = key_valid[:, None, None, :] if key_valid is not None else None
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(y.transpose(1, 2).reshape(b, nq, dim))


class _Layer(nn.Module):
    """所有残差分支采用 Pre-Norm；FFN 宽度为 D→4D→2D→D。"""

    def __init__(self, dim: int, heads: int, theta: float) -> None:
        super().__init__()
        self.norm_ca_q = RMSNormTokens(dim)
        self.norm_ca_kv = RMSNormTokens(dim)
        self.ca = _Attention(dim, heads, theta)
        self.norm_sa = RMSNormTokens(dim)
        self.sa = _Attention(dim, heads, theta)
        self.norm_ffn = RMSNormTokens(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), SwiGLU(dim=-1),
                                 nn.Linear(2 * dim, dim))

    def forward(self, tokens, image, positions, ca_positions, image_positions, query_hint,
                image_hint, image_valid, lidar=None):
        tokens = tokens + self.ca(self.norm_ca_q(tokens), self.norm_ca_kv(image),
                                  ca_positions, image_positions, query_hint, image_hint,
                                  image_valid)
        if lidar is not None:
            tokens = lidar(tokens)
        h = self.norm_sa(tokens)
        tokens = tokens + self.sa(h, h, positions, positions)
        return tokens + self.ffn(self.norm_ffn(tokens))


class DrivingTransformer(nn.Module):
    """共享 BEV Patch 与 Detect Token 的六层感知编码器。"""

    def __init__(self, cfg) -> None:
        super().__init__()
        dim = cfg.work_dim
        bev, det, fr = cfg.bev, cfg.detection, cfg.frustum
        self.bev_shape = (bev.height, bev.width)
        self.num_bev = bev.height * bev.width
        self.scale = fr.coord_symlog_scale
        self.z_min = bev.z_min_m
        self.z_max = bev.z_max_m
        self.x_min, self.x_span = bev.x_min_m, bev.x_max_m - bev.x_min_m
        self.y_min, self.y_span = bev.y_min_m, bev.y_max_m - bev.y_min_m
        self.n_depth = self._depth_count(fr)
        self.bev_tokens = nn.Parameter(torch.empty(1, self.num_bev, dim))
        nn.init.normal_(self.bev_tokens, std=cfg.bev_encoder.register_init_std)
        self.detect_tokens = nn.Parameter(torch.empty(1, det.num_queries, dim))
        nn.init.normal_(self.detect_tokens, std=cfg.bev_encoder.register_init_std)
        self.anchors = nn.Parameter(torch.rand(det.num_queries, 3))
        coords = self._bev_centers(bev)
        self.register_buffer("bev_centers", coords, persistent=False)
        ray_dim = 5 * self.n_depth * 3
        mlp = nn.Sequential(nn.Linear(ray_dim, fr.mlp_hidden), nn.SiLU(),
                            nn.Linear(fr.mlp_hidden, dim))
        self.query_ray_mlp = mlp
        self.key_ray_mlp = copy.deepcopy(mlp)
        self.layers = nn.ModuleList(_Layer(dim, cfg.attention.num_heads,
                                           cfg.bev_encoder.rope_theta) for _ in range(6))

    @staticmethod
    def _depth_count(fr):
        depth, count = fr.depth_min_m, 0
        while depth <= fr.depth_max_m:
            count += 1
            fraction = (depth - fr.depth_min_m) / (fr.depth_max_m - fr.depth_min_m)
            depth += fr.step_near_m + (fr.step_far_m - fr.step_near_m) * fraction
        return count

    @staticmethod
    def _bev_centers(bev):
        x = bev.x_max_m - (torch.arange(bev.height) + .5) * (
            bev.x_max_m - bev.x_min_m) / bev.height
        y = bev.y_min_m + (torch.arange(bev.width) + .5) * (
            bev.y_max_m - bev.y_min_m) / bev.width
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        return torch.stack((xx, yy, torch.zeros_like(xx)), -1).reshape(-1, 3)

    def _query_geometry(self, batch, device):
        bev = self.bev_centers.to(device)
        anchor = self.anchors.sigmoid()
        anchor = torch.stack((self.x_min + anchor[:, 0] * self.x_span,
                              self.y_min + anchor[:, 1] * self.y_span,
                              self.z_min + anchor[:, 2] * (self.z_max - self.z_min)), -1)
        xyz = torch.cat((bev, anchor), 0).expand(batch, -1, -1)
        # 查询方同样使用 5×N×3 格式，以便两个射线 MLP 严格同构。
        z = torch.linspace(self.z_min, self.z_max, self.n_depth, device=device)
        rays = xyz[:, :, None, None, :].expand(-1, -1, 5, self.n_depth, -1).clone()
        rays[:, :self.num_bev, :, :, 2] = z
        return xyz, rays

    def forward(self, image, image_rays, image_time, image_valid=None, lidar_fusers=None):
        """返回末层 BEV、每层 BEV 与 Detect 分层 Token。"""
        check_transformer_inputs(image, image_rays, image_time, image_valid,
                                 self.bev_tokens.shape[-1], self.n_depth)
        batch, length, dim = image.shape
        xyz, query_rays = self._query_geometry(batch, image.device)
        tokens = torch.cat((self.bev_tokens, self.detect_tokens), 1).expand(batch, -1, -1)
        query_hint = self.query_ray_mlp(_symlog(query_rays.float(), self.scale).flatten(2))
        image_hint = self.key_ray_mlp(_symlog(image_rays.float(), self.scale).flatten(2))
        q_pos = torch.cat((xyz[..., :2], torch.zeros_like(xyz[..., 2:])), dim=-1)
        k_pos = torch.cat((image_rays[:, :, 0, 0, :2], image_time.unsqueeze(-1)), dim=-1)
        ca_q_pos = torch.cat((_symlog(q_pos[..., :2], self.scale), q_pos[..., 2:]), dim=-1)
        ca_k_pos = torch.cat((_symlog(k_pos[..., :2], self.scale), k_pos[..., 2:]), dim=-1)
        bev_layers, detect_layers = [], []
        for index, layer in enumerate(self.layers):
            fuse = None if lidar_fusers is None or index not in (0, 2, 4) else lidar_fusers[index // 2]
            tokens = layer(tokens, image, q_pos, ca_q_pos, ca_k_pos, query_hint, image_hint,
                           image_valid, fuse)
            bev_layers.append(tokens[:, :self.num_bev].transpose(1, 2).reshape(
                batch, dim, *self.bev_shape))
            detect_layers.append(tokens[:, self.num_bev:])
        return bev_layers[-1], tuple(bev_layers), tuple(detect_layers), xyz[:, self.num_bev:]
