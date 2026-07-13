"""目标点嵌入层：把 ego 坐标系目标点编码为目标导航特征图。

模块: model/target_point_embedding/target_point_embedding.py
依赖: torch, config.schema.TargetPointEmbeddingCfg, model.residual_block.ResidualBlock,
      model.target_point_embedding.checks.target_point_embedding_checks
读取配置:
    model.target_point_embedding.coordinate_dim
    model.target_point_embedding.grid_height / grid_width
    model.target_point_embedding.x_min_m / x_max_m / y_min_m / y_max_m
    model.target_point_embedding.vector_order
    model.target_point_embedding.stem_channels
    model.target_point_embedding.stem_kernel_size / stem_stride / stem_padding
    model.target_point_embedding.num_residual_blocks
    model.target_point_embedding.output_channels
    model.target_point_embedding.output_height / output_width
对外接口:
    - TargetPointEmbedding(cfg) -> nn.Module   # 输入 [B,2] ego 目标点，输出 [B, output_channels, output_height, output_width]
说明: 目标点先转为覆盖车前后/左右的栅格向量场，与栅格中心坐标各自 Symlog 后沿通道拼接（[B,4,H,W]）；
      再经 16×16 卷积 16 倍降采样升到 stem_channels，过 num_residual_blocks 层 2D 残差块，末端 1×1 卷积
      升到 output_channels，输出目标导航特征图。
      精度边界（规范：混精外置，同 perception_model）：向量场与 Symlog 计算恒在 FP32（关闭 autocast），
      其后卷积/残差块段在 BF16 autocast 下运行；autocast 设备类型由输入张量推导，meta/不支持设备回退空
      上下文。参数与其约束的唯一来源为 config；枚举与降采样形状推导在 config/schema.py 加载期一次性校验，
      本文件不再重复（规范 §6/§7.3）；运行期仅校验入参张量与输出形状。vector_transform 为 schema 约束的
      契约键，当前仅 symlog，实现按该唯一值处理、运行期不读取（故不列入「读取配置」，避免文件头与代码不符）。
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn

from config.schema import TargetPointEmbeddingCfg
from model.residual_block import ResidualBlock
from model.target_point_embedding.checks.target_point_embedding_checks import (
    check_output_features,
    check_target_points,
)


__all__ = ["TargetPointEmbedding"]

# 卷积/残差段的低精度：任务要求 Symlog 与向量计算走 FP32、其余走 BF16（设计常量，非实验参数）
_LOW_PRECISION = torch.bfloat16


class TargetPointEmbedding(nn.Module):
    """将目标点编码为目标导航特征图。

    Args:
        cfg: 目标点嵌入层配置，唯一来源为 `config.model.target_point_embedding`。

    Shape:
        输入: `[B, 2]`，ego 坐标系目标点，单位 meter，坐标为 `[x, y]`。
        输出: `[B, output_channels, output_height, output_width]`。
    """

    def __init__(self, cfg: TargetPointEmbeddingCfg) -> None:
        super().__init__()
        self.cfg = cfg
        # grid_xy 恒以 FP32 存储：Symlog 精度依赖它，即使外部误将模块降精也在前向内显式转回 FP32
        self.register_buffer("grid_xy", _build_grid_xy(cfg))
        # 输入通道 = 向量场(coordinate_dim) ⊕ 栅格中心坐标(coordinate_dim)
        self.stem = nn.Conv2d(
            in_channels=cfg.coordinate_dim * 2,
            out_channels=cfg.stem_channels,
            kernel_size=tuple(cfg.stem_kernel_size),
            stride=tuple(cfg.stem_stride),
            padding=tuple(cfg.stem_padding),
        )
        self.res_blocks = nn.Sequential(
            *(ResidualBlock(cfg.stem_channels) for _ in range(cfg.num_residual_blocks))
        )
        self.output_proj = nn.Conv2d(cfg.stem_channels, cfg.output_channels, kernel_size=1)

    def forward(self, target_points: torch.Tensor) -> torch.Tensor:
        """输出目标导航特征图。

        `target_points` 为 ego 坐标系米制 `[x, y]`，shape `[B, 2]`。向量场与 Symlog 恒在 FP32，
        随后 16×16 降采样卷积、残差块与 1×1 升维在 BF16 autocast 下运行。
        """

        check_target_points(target_points, self.cfg.coordinate_dim)
        # FP32 段：向量场 + Symlog + 栅格坐标 Symlog 拼接（关闭 autocast，恒 FP32）
        with self._autocast(target_points.device, enabled=False):
            input_field = self._build_input_field(target_points.to(dtype=torch.float32))
        # BF16 段：降采样卷积 → 残差块 → 1×1 升维（主库参数仍为 FP32，autocast 只降算子精度）
        with self._autocast(target_points.device, enabled=True):
            features = self.output_proj(self.res_blocks(self.stem(input_field)))
        check_output_features(
            features,
            self.cfg.output_channels,
            self.cfg.output_height,
            self.cfg.output_width,
        )
        return features

    def _build_input_field(self, target_points_fp32: torch.Tensor) -> torch.Tensor:
        """构造送入卷积的输入场：Symlog(向量场) ⊕ Symlog(栅格中心坐标) → `[B, 2C, H, W]`（FP32）。"""

        grid_xy = self.grid_xy.to(device=target_points_fp32.device, dtype=torch.float32)
        vector_field = self._build_meter_vector_field(target_points_fp32, grid_xy)
        # 目标点向量场与栅格中心坐标各自 Symlog，再沿通道拼接：既给出到目标的相对位移，也给出绝对栅格坐标
        vector_symlog = _symlog(vector_field)                       # [B, H, W, 2]
        grid_symlog = _symlog(grid_xy).unsqueeze(0).expand_as(vector_symlog)  # [B, H, W, 2]
        # [B, H, W, 2C] -> [B, 2C, H, W]
        return torch.cat((vector_symlog, grid_symlog), dim=-1).permute(0, 3, 1, 2).contiguous()

    def _build_meter_vector_field(
        self, target_points_fp32: torch.Tensor, grid_xy: torch.Tensor
    ) -> torch.Tensor:
        """目标点到栅格中心的米制向量场：`[B, 2]` -> `[B, H, W, 2]`。"""

        # vector_order 仅取 grid_minus_target / target_minus_grid（schema 已拦截），此处二选一。
        if self.cfg.vector_order == "grid_minus_target":
            return grid_xy.unsqueeze(0) - target_points_fp32[:, None, None, :]
        return target_points_fp32[:, None, None, :] - grid_xy.unsqueeze(0)

    def _autocast(self, device: torch.device, enabled: bool) -> Any:
        """构造 autocast 上下文：enabled 时用 BF16，否则关闭；meta/不支持设备回退空上下文。"""

        if device.type == "meta":
            return nullcontext()
        try:
            return torch.autocast(device_type=device.type, dtype=_LOW_PRECISION, enabled=enabled)
        except (RuntimeError, ValueError):
            return nullcontext()


def _symlog(x: torch.Tensor) -> torch.Tensor:
    """Symlog 变换 sign(x)·log1p(|x|)，压缩大位移/大坐标的动态范围（由调用方保证 FP32）。"""

    return torch.sign(x) * torch.log1p(torch.abs(x))


def _build_grid_xy(cfg: TargetPointEmbeddingCfg) -> torch.Tensor:
    x_cell_size = (cfg.x_max_m - cfg.x_min_m) / float(cfg.grid_height)
    y_cell_size = (cfg.y_max_m - cfg.y_min_m) / float(cfg.grid_width)
    x_positions = cfg.x_min_m + (torch.arange(cfg.grid_height, dtype=torch.float32) + 0.5) * x_cell_size
    y_positions = cfg.y_min_m + (torch.arange(cfg.grid_width, dtype=torch.float32) + 0.5) * y_cell_size
    grid_x, grid_y = torch.meshgrid(x_positions, y_positions, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1)
