"""BEV 查询几何嵌入：仅把 BEV 栅格中心 xyz（含垂直 z 采样）编码为初始查询网格。

模块: model/bev_query_embedding/bev_query_embedding.py
依赖: torch, contextlib, model.bev_query_embedding.checks.bev_query_embedding_checks
读取配置: —（几何量程/分辨率/z 采样/尺度/mlp_hidden 由调用方传入，来源 config.model.driving）
对外接口:
    - BevQueryEmbedding(out_dim, x_min_m, x_max_m, y_min_m, y_max_m, height, width,
                        z_min_m, z_max_m, z_step_m, coord_symlog_scale, mlp_hidden) -> nn.Module
        forward(batch_size, device, grid_xy=None) -> Tensor
说明: 默认把每个 BEV cell 中心 xy 扩展为一列 xyz，也可编码逐 batch 刚性变换后的实际网格。
      几何先经 Symlog 归一和逐点 MLP，再沿 z 均值聚合；查询初始化不接收、不编码目标点。
      栅格/Symlog 恒 FP32，逐点 MLP 在 BF16 autocast 下运行。
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn

from model.bev_query_embedding.checks.bev_query_embedding_checks import (
    check_bev_query_args,
    check_bev_query_inputs,
)


__all__ = ["BevQueryEmbedding"]

_LOW_PRECISION = torch.bfloat16


class BevQueryEmbedding(nn.Module):
    """把默认或实际 BEV 栅格 xyz 编码为初始查询网格。

    形状:
        可选 grid_xy: `[B, height, width, 2]`；
        输出: `[B, out_dim, height, width]`。
    """

    def __init__(self, out_dim: int, x_min_m: float, x_max_m: float, y_min_m: float, y_max_m: float,
                 height: int, width: int, z_min_m: float, z_max_m: float, z_step_m: float,
                 coord_symlog_scale: float, mlp_hidden: int) -> None:
        super().__init__()
        check_bev_query_args(out_dim, height, width, mlp_hidden, coord_symlog_scale)
        self.out_dim = out_dim
        self.height = height
        self.width = width
        self.coord_symlog_scale = coord_symlog_scale
        self.register_buffer("grid_xy", _build_grid_xy(
            x_min_m, x_max_m, y_min_m, y_max_m, height, width))
        self.register_buffer("z_samples", _build_z_samples(z_min_m, z_max_m, z_step_m))
        self.mlp = nn.Sequential(
            nn.Linear(3, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, out_dim),
        )

    def forward(self, batch_size: int, device: torch.device,
                grid_xy: torch.Tensor = None) -> torch.Tensor:
        """编码默认或外部实际网格，返回纯几何 BEV 查询。"""
        check_bev_query_inputs(batch_size, device, grid_xy, self.height, self.width)
        if grid_xy is None:
            grid_xy = self.grid_xy[None].expand(batch_size, -1, -1, -1)
        with self._autocast(device, enabled=False):
            columns = self._build_columns(grid_xy.float().to(device))
        with self._autocast(device, enabled=True):
            query = self.mlp(columns).mean(dim=3)
        return query.permute(0, 3, 1, 2).reshape(
            batch_size, self.out_dim, self.height, self.width)

    def _build_columns(self, grid_xy: torch.Tensor) -> torch.Tensor:
        """构造逐 cell、逐高度的三维几何向量并作 Symlog 归一。"""
        batch_size = int(grid_xy.shape[0])
        z = self.z_samples.to(grid_xy.device)
        nz = int(z.shape[0])
        xy = grid_xy[:, :, :, None, :].expand(
            batch_size, self.height, self.width, nz, 2)
        z_column = z[None, None, None, :, None].expand(
            batch_size, self.height, self.width, nz, 1)
        columns = torch.cat((xy, z_column), dim=-1)
        return torch.sign(columns) * torch.log1p(columns.abs()) * self.coord_symlog_scale

    def _autocast(self, device: torch.device, enabled: bool) -> Any:
        """构造 autocast 上下文；meta 或不支持设备回退为空上下文。"""
        if device.type == "meta":
            return nullcontext()
        try:
            return torch.autocast(device_type=device.type, dtype=_LOW_PRECISION, enabled=enabled)
        except (RuntimeError, ValueError):
            return nullcontext()


def _build_grid_xy(x_min: float, x_max: float, y_min: float, y_max: float,
                   height: int, width: int) -> torch.Tensor:
    """生成 BEV cell 中心；行 0 为远端，末行为近端。"""
    x_cell = (x_max - x_min) / float(height)
    y_cell = (y_max - y_min) / float(width)
    xs = x_max - (torch.arange(height, dtype=torch.float32) + 0.5) * x_cell
    ys = y_min + (torch.arange(width, dtype=torch.float32) + 0.5) * y_cell
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    return torch.stack((gx, gy), dim=-1)


def _build_z_samples(z_min: float, z_max: float, z_step: float) -> torch.Tensor:
    """生成包含两端点的垂直 z 采样。"""
    count = int(round((z_max - z_min) / z_step)) + 1
    return z_min + torch.arange(count, dtype=torch.float32) * z_step
