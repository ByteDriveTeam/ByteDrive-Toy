"""目标点嵌入层：把 ego 坐标系目标点编码为目标导航点 Token。

模块: model/target_point_embedding.py
依赖: torch, config.schema.TargetPointEmbeddingCfg, model.target_point_embedding_checks
读取配置:
    model.target_point_embedding.coordinate_dim
    model.target_point_embedding.grid_height / grid_width
    model.target_point_embedding.x_min_m / x_max_m / y_min_m / y_max_m
    model.target_point_embedding.vector_order
    model.target_point_embedding.feature_channels
    model.target_point_embedding.conv1_kernel_size / conv1_stride / conv1_padding
    model.target_point_embedding.conv2_kernel_size / conv2_stride / conv2_padding
    model.target_point_embedding.downsample_kernel_size / downsample_stride / downsample_padding
    model.target_point_embedding.output_height / output_width
    model.target_point_embedding.goal_token_count / hidden_dim
对外接口:
    - TargetPointEmbedding(cfg) -> nn.Module   # 输入 [B,2] ego 目标点，输出 [B, goal_token_count, hidden_dim]
说明: 目标点先转为覆盖车前后/左右的栅格向量场，Symlog 变换后经三层卷积下采样，展平投影为目标导航点
      Token；向量场、卷积与投影全程强制 FP32。参数与其约束的唯一来源为 config：枚举与卷积链形状
      推导在 config/schema.py 加载期一次性校验，本文件不再重复（规范 §6/§7.3）；运行期仅校验入参张量。
      vector_transform / flatten_order / dtype 为 schema 约束的契约键，当前各仅一个支持值，实现按该
      唯一值处理、运行期不读取（故不列入「读取配置」，避免文件头与代码不符）。
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn

from config.schema import TargetPointEmbeddingCfg
from model.target_point_embedding_checks import check_embedded_features, check_target_points


__all__ = ["TargetPointEmbedding"]


class TargetPointEmbedding(nn.Module):
    """将目标点编码为目标导航点 Token。

    Args:
        cfg: 目标点嵌入层配置，唯一来源为 `config.model.target_point_embedding`。

    Shape:
        输入: `[B, 2]`，ego 坐标系目标点，单位 meter，坐标为 `[x, y]`。
        输出: `[B, goal_token_count, hidden_dim]`。
    """

    def __init__(self, cfg: TargetPointEmbeddingCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("grid_xy", _build_grid_xy(cfg))
        self.conv1 = nn.Conv2d(
            in_channels=cfg.coordinate_dim,
            out_channels=cfg.feature_channels,
            kernel_size=tuple(cfg.conv1_kernel_size),
            stride=tuple(cfg.conv1_stride),
            padding=tuple(cfg.conv1_padding),
        )
        self.conv2 = nn.Conv2d(
            in_channels=cfg.feature_channels,
            out_channels=cfg.feature_channels,
            kernel_size=tuple(cfg.conv2_kernel_size),
            stride=tuple(cfg.conv2_stride),
            padding=tuple(cfg.conv2_padding),
        )
        self.downsample = nn.Conv2d(
            in_channels=cfg.feature_channels,
            out_channels=cfg.feature_channels,
            kernel_size=tuple(cfg.downsample_kernel_size),
            stride=tuple(cfg.downsample_stride),
            padding=tuple(cfg.downsample_padding),
        )
        flattened_dim = cfg.feature_channels * cfg.output_height * cfg.output_width
        projected_dim = cfg.goal_token_count * cfg.hidden_dim
        self.output_projection = nn.Linear(flattened_dim, projected_dim)
        _force_floating_tensors_to_float32(self)

    def _apply(self, fn: Any) -> "TargetPointEmbedding":
        super()._apply(fn)
        _force_floating_tensors_to_float32(self)
        return self

    def forward(self, target_points: torch.Tensor) -> torch.Tensor:
        """输出目标导航点 Token。

        `target_points` 为 ego 坐标系米制 `[x, y]`，shape `[B, 2]`。目标点到栅格中心的
        米制向量先做 Symlog 变换再进入卷积；向量场、卷积与投影全程 FP32。
        """

        check_target_points(target_points, self.cfg.coordinate_dim)
        with _disabled_autocast(self.grid_xy):
            target_points_fp32 = target_points.to(dtype=torch.float32)
            vector_features = self._build_vector_features(target_points_fp32)
            embedded_features = self.downsample(self.conv2(self.conv1(vector_features)))
            check_embedded_features(
                embedded_features,
                self.cfg.feature_channels,
                self.cfg.output_height,
                self.cfg.output_width,
            )
            # flatten_order 仅支持 channel_height_width（schema 已拦截），[B,C,H,W] -> [B,C*H*W]
            flattened_features = embedded_features.reshape(int(embedded_features.shape[0]), -1)
            projected_tokens = self.output_projection(flattened_features)
            return projected_tokens.reshape(
                int(target_points.shape[0]),
                self.cfg.goal_token_count,
                self.cfg.hidden_dim,
            )

    def _build_vector_features(self, target_points_fp32: torch.Tensor) -> torch.Tensor:
        """构造送入卷积的向量场：米制向量场 → Symlog → `[B, 2, H, W]`。"""

        meter_vector_field = self._build_meter_vector_field(target_points_fp32)
        # vector_transform 仅支持 symlog（schema 已拦截）：sign(v) * log1p(|v|)，压缩大位移动态范围。
        normalized_vector_field = torch.sign(meter_vector_field) * torch.log1p(
            torch.abs(meter_vector_field)
        )
        # [B, H, W, 2] -> [B, 2, H, W]
        return normalized_vector_field.permute(0, 3, 1, 2).contiguous()

    def _build_meter_vector_field(self, target_points_fp32: torch.Tensor) -> torch.Tensor:
        """目标点到栅格中心的米制向量场：`[B, 2]` -> `[B, H, W, 2]`。"""

        grid_xy = self.grid_xy.to(device=target_points_fp32.device, dtype=torch.float32)
        # vector_order 仅取 grid_minus_target / target_minus_grid（schema 已拦截），此处二选一。
        if self.cfg.vector_order == "grid_minus_target":
            return grid_xy.unsqueeze(0) - target_points_fp32[:, None, None, :]
        return target_points_fp32[:, None, None, :] - grid_xy.unsqueeze(0)


def _build_grid_xy(cfg: TargetPointEmbeddingCfg) -> torch.Tensor:
    x_cell_size = (cfg.x_max_m - cfg.x_min_m) / float(cfg.grid_height)
    y_cell_size = (cfg.y_max_m - cfg.y_min_m) / float(cfg.grid_width)
    x_positions = cfg.x_min_m + (torch.arange(cfg.grid_height, dtype=torch.float32) + 0.5) * x_cell_size
    y_positions = cfg.y_min_m + (torch.arange(cfg.grid_width, dtype=torch.float32) + 0.5) * y_cell_size
    grid_x, grid_y = torch.meshgrid(x_positions, y_positions, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1)


def _disabled_autocast(reference_tensor: torch.Tensor) -> Any:
    """根据参考张量设备构造禁用 autocast 的上下文，保证内部计算恒在 FP32。"""

    if reference_tensor.device.type == "meta":
        return nullcontext()
    try:
        return torch.autocast(device_type=reference_tensor.device.type, enabled=False)
    except (RuntimeError, ValueError):
        return nullcontext()


def _force_floating_tensors_to_float32(module: nn.Module) -> None:
    """将模块内所有浮点参数、buffer 和已有梯度恢复为 FP32（对抗外部 .half()/.to() 下溢）。"""

    with torch.no_grad():
        for parameter in module.parameters(recurse=True):
            if parameter.is_floating_point() and parameter.dtype != torch.float32:
                parameter.data = parameter.data.to(dtype=torch.float32)
            if parameter.grad is not None and parameter.grad.is_floating_point():
                parameter.grad.data = parameter.grad.data.to(dtype=torch.float32)

        for buffer in module.buffers(recurse=True):
            if buffer.is_floating_point() and buffer.dtype != torch.float32:
                buffer.data = buffer.data.to(dtype=torch.float32)
