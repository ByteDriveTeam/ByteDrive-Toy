"""目标点嵌入层：把 BEV 栅格中心 xyz 坐标（含垂直 z 采样）与目标点相对向量编码为初始 BEV 查询网格。

模块: model/target_point_embedding/target_point_embedding.py
依赖: torch, contextlib, model.target_point_embedding.checks.target_point_embedding_checks
读取配置: —（几何量程/分辨率/z 采样/尺度/mlp_hidden/vector_order 由调用方传入，来源 config.model.driving）
对外接口:
    - TargetPointEmbedding(out_dim, x_min_m, x_max_m, y_min_m, y_max_m, height, width,
                           z_min_m, z_max_m, z_step_m, coord_symlog_scale, mlp_hidden,
                           vector_order) -> nn.Module
        forward(target_points) -> Tensor   # [B,2] ego 目标点 → [B, out_dim, height, width] 初始 BEV 查询
说明: 每个 BEV cell 中心 xy 扩展为一列 xyz（z 取 [z_min,z_max]@z_step 多值），逐 (cell,z) 拼上「到目标点的
      相对位移(xy)」构成 5 维几何向量，Symlog 归一到[-1,1]后过 Linear→SiLU→Linear 逐点编码，再沿 z 维
      聚合（均值）为该 cell 的 out_dim 维查询特征。垂直 z 多值经 MLP 非线性后聚合，使高度维带来的几何先验
      不被线性抵消；相对目标向量注入导航意图。自车位于 BEV 下方中心（x 前向、y 右向；行 0 对应 x_min）。
      精度边界（混精外置，同 perception_model / frustum_encoding）：栅格/Symlog 几何恒 FP32（关闭 autocast），
      逐点 MLP 在 BF16 autocast 下运行；meta/不支持设备回退空上下文。
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn

from model.target_point_embedding.checks.target_point_embedding_checks import (
    check_bev_query_args,
    check_target_points,
)


__all__ = ["TargetPointEmbedding"]

_LOW_PRECISION = torch.bfloat16


class TargetPointEmbedding(nn.Module):
    """把 BEV 栅格 xyz + 目标点相对向量编码为初始 BEV 查询网格。

    Shape:
        输入: `[B, 2]`，ego 坐标系目标点（米，[x, y]）。
        输出: `[B, out_dim, height, width]`，初始 BEV 查询。
    """

    def __init__(self, out_dim: int, x_min_m: float, x_max_m: float, y_min_m: float, y_max_m: float,
                 height: int, width: int, z_min_m: float, z_max_m: float, z_step_m: float,
                 coord_symlog_scale: float, mlp_hidden: int, vector_order: str) -> None:
        super().__init__()
        check_bev_query_args(out_dim, height, width, mlp_hidden, coord_symlog_scale, vector_order)
        self.out_dim = out_dim
        self.height = height
        self.width = width
        self.coord_symlog_scale = coord_symlog_scale
        self.vector_order = vector_order
        # 栅格中心 xy（FP32 常量）与垂直 z 采样
        self.register_buffer("grid_xy", _build_grid_xy(x_min_m, x_max_m, y_min_m, y_max_m, height, width))
        self.register_buffer("z_samples", _build_z_samples(z_min_m, z_max_m, z_step_m))
        # 逐 (cell, z) 输入维 = xyz(3) ⊕ 到目标点相对位移 xy(2)
        self.mlp = nn.Sequential(
            nn.Linear(5, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, out_dim),
        )

    def forward(self, target_points: torch.Tensor) -> torch.Tensor:
        """ego 目标点 → 初始 BEV 查询 `[B, out_dim, height, width]`。"""
        check_target_points(target_points)
        device = target_points.device
        b = int(target_points.shape[0])
        with self._autocast(device, enabled=False):
            columns = self._build_columns(target_points.float())  # [B, H, W, Nz, 5]
        with self._autocast(device, enabled=True):
            encoded = self.mlp(columns)          # [B, H, W, Nz, out_dim]
            query = encoded.mean(dim=3)          # 沿 z 聚合 → [B, H, W, out_dim]
        return query.permute(0, 3, 1, 2).reshape(b, self.out_dim, self.height, self.width)

    def _build_columns(self, target_points: torch.Tensor) -> torch.Tensor:
        """构造逐 (cell, z) 的 5 维几何向量并 Symlog 归一：`[B, H, W, Nz, 5]`（FP32）。"""
        grid_xy = self.grid_xy.to(target_points.device)          # [H,W,2]
        z = self.z_samples.to(target_points.device)              # [Nz]
        h, w = self.height, self.width
        nz = int(z.shape[0])
        # 到目标点相对位移（xy，与 z 无关）
        if self.vector_order == "grid_minus_target":
            rel = grid_xy[None] - target_points[:, None, None, :]  # [B,H,W,2]
        else:
            rel = target_points[:, None, None, :] - grid_xy[None]
        b = target_points.shape[0]
        # 广播拼出 [B,H,W,Nz,5] = (x,y,z, rel_x, rel_y)
        xy = grid_xy[None, :, :, None, :].expand(b, h, w, nz, 2)
        zc = z[None, None, None, :, None].expand(b, h, w, nz, 1)
        rel = rel[:, :, :, None, :].expand(b, h, w, nz, 2)
        columns = torch.cat((xy, zc, rel), dim=-1)
        return torch.sign(columns) * torch.log1p(columns.abs()) * self.coord_symlog_scale

    def _autocast(self, device: torch.device, enabled: bool) -> Any:
        """构造 autocast 上下文：几何 FP32、MLP BF16；meta/不支持设备回退空上下文。"""
        if device.type == "meta":
            return nullcontext()
        try:
            return torch.autocast(device_type=device.type, dtype=_LOW_PRECISION, enabled=enabled)
        except (RuntimeError, ValueError):
            return nullcontext()


def _build_grid_xy(x_min: float, x_max: float, y_min: float, y_max: float,
                   height: int, width: int) -> torch.Tensor:
    """BEV 栅格中心 [H,W,2]：行(H)沿 x 前向、列(W)沿 y 右向。

    行约定与 data.driving_targets / ego_xy_to_pixel 一致：行 0 = 远 x_max（BEV 上沿），末行 = 近 x_min
    （自车在下沿中心），使模型 BEV 与监督场/轨迹渲染同向。
    """
    x_cell = (x_max - x_min) / float(height)
    y_cell = (y_max - y_min) / float(width)
    xs = x_max - (torch.arange(height, dtype=torch.float32) + 0.5) * x_cell
    ys = y_min + (torch.arange(width, dtype=torch.float32) + 0.5) * y_cell
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    return torch.stack((gx, gy), dim=-1)


def _build_z_samples(z_min: float, z_max: float, z_step: float) -> torch.Tensor:
    """垂直 z 采样 [z_min, z_max]@z_step（含端点），供每 cell 垂直列编码。"""
    n = int(round((z_max - z_min) / z_step)) + 1
    return z_min + torch.arange(n, dtype=torch.float32) * z_step
