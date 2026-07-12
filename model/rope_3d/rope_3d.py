"""通用 3D RoPE 旋转位置编码。

模块: model/rope_3d/rope_3d.py
依赖: torch, model.rope_3d.checks.rope_3d_checks
读取配置: —（axis_dims 与 theta 由调用方以参数传入，自身不读 config）
对外接口:
    - apply_rope_3d(features, positions, axis_dims, theta) -> Tensor   # 对特征施加 3D RoPE，输出 FP32
    - RoPE3D(axis_dims, theta) -> nn.Module                            # 对 query/key 施加同一 3D RoPE
说明: 只消费调用方传入的三维位置坐标，不生成网格/中心化/归一化/选头；旋转全程 FP32。
      入参校验下沉到 rope_3d_checks；_align_angles_to_features 内部对「计算出的中间 shape」
      的少量断言保留在本体（规范 §7.1 允许少而精的内联断言）。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from model.rope_3d.checks.rope_3d_checks import (
    check_axis_dims,
    check_query_key_match,
    check_rope_capacity,
    check_rope_inputs,
    check_theta,
)


__all__ = ["RoPE3D", "apply_rope_3d"]


def apply_rope_3d(
    features: torch.Tensor,
    positions: torch.Tensor,
    axis_dims: Sequence[int],
    theta: float,
) -> torch.Tensor:
    """对特征张量应用 3D RoPE。

    本函数只消费调用方传入的三维位置坐标，不生成网格坐标，也不做中心化、
    归一化或头选择。位置坐标最后一维的 3 个值按调用方约定解释，例如视觉
    Token 可传入 `[H, W, T]`。

    Args:
        features: 待旋转特征，最后两维为 token 和通道。
        positions: 已由调用方准备好的三维位置坐标。
        axis_dims: 三个坐标轴各自占用的 rotary 通道数，每项必须为正偶数。
        theta: RoPE 基频。

    Returns:
        应用 3D RoPE 后的 FP32 特征，shape 与 `features` 相同。

    Shape:
        `features`: `[..., N, C]`
        `positions`: `[N, 3]` 或 `[..., N, 3]`
        输出: `[..., N, C]`
    """

    check_rope_inputs(features, positions)
    validated_axis_dims = check_axis_dims(axis_dims)
    feature_dim = int(features.shape[-1])
    check_rope_capacity(sum(validated_axis_dims), feature_dim)
    check_theta(theta)

    features_fp32 = features.to(dtype=torch.float32)
    positions_fp32 = positions.to(dtype=torch.float32)
    rotated_parts = []
    cursor = 0
    # 逐轴切片 + 累进 cursor 有副作用且各轴通道宽度不同，for 比推导式更清晰（规范 §9 第 4 档）。
    for axis_index, axis_dim in enumerate(validated_axis_dims):
        next_cursor = cursor + axis_dim
        axis_features = features_fp32[..., cursor:next_cursor]
        axis_positions = positions_fp32[..., axis_index]
        rotated_parts.append(_apply_1d_rope(axis_features, axis_positions, axis_dim, theta))
        cursor = next_cursor

    if cursor < feature_dim:
        rotated_parts.append(features_fp32[..., cursor:])
    return torch.cat(rotated_parts, dim=-1)


class RoPE3D(nn.Module):
    """通用 3D RoPE 模块。

    Args:
        axis_dims: 三个坐标轴各自占用的 rotary 通道数，每项必须为正偶数。
        theta: RoPE 基频。

    Shape:
        `query`: `[..., N, C]`
        `key`: `[..., N, C]`
        `positions`: `[N, 3]` 或 `[..., N, 3]`
        输出: 两个 shape 与输入一致的 FP32 张量。
    """

    def __init__(self, axis_dims: Sequence[int], theta: float) -> None:
        super().__init__()
        self.axis_dims = check_axis_dims(axis_dims)
        check_theta(theta)
        self.theta = float(theta)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """对 query 和 key 应用相同的 3D RoPE。"""

        check_query_key_match(query, key)
        rotated_query = apply_rope_3d(query, positions, self.axis_dims, self.theta)
        rotated_key = apply_rope_3d(key, positions, self.axis_dims, self.theta)
        return rotated_query, rotated_key


def _apply_1d_rope(
    axis_features: torch.Tensor,
    axis_positions: torch.Tensor,
    axis_dim: int,
    theta: float,
) -> torch.Tensor:
    pair_count = axis_dim // 2
    frequency_indices = torch.arange(
        pair_count,
        device=axis_positions.device,
        dtype=torch.float32,
    )
    inv_frequencies = torch.pow(
        torch.tensor(float(theta), device=axis_positions.device, dtype=torch.float32),
        -2.0 * frequency_indices / float(axis_dim),
    )
    angles = axis_positions[..., None] * inv_frequencies
    angles = _align_angles_to_features(angles, axis_features)
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)

    even_features = axis_features[..., 0::2]
    odd_features = axis_features[..., 1::2]
    rotated_even = even_features * cos_angles - odd_features * sin_angles
    rotated_odd = even_features * sin_angles + odd_features * cos_angles
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


def _align_angles_to_features(angles: torch.Tensor, axis_features: torch.Tensor) -> torch.Tensor:
    feature_prefix_ndim = axis_features.ndim - 2
    angle_prefix_ndim = angles.ndim - 2
    missing_prefix_ndim = feature_prefix_ndim - angle_prefix_ndim
    if missing_prefix_ndim < 0:
        raise ValueError(
            "positions 的前缀维度不能多于 features 的前缀维度，"
            f"实际分别为 {angle_prefix_ndim} 和 {feature_prefix_ndim}。"
        )
    if missing_prefix_ndim == 0:
        return angles

    prefix_shape = tuple(angles.shape[:-2])
    feature_prefix_shape = tuple(axis_features.shape[:-2])
    token_and_pair_shape = tuple(angles.shape[-2:])
    batch_aligned_prefix = (*prefix_shape, *((1,) * missing_prefix_ndim))
    if _is_broadcastable(batch_aligned_prefix, feature_prefix_shape):
        return angles.reshape(*batch_aligned_prefix, *token_and_pair_shape)

    trailing_aligned_prefix = (*((1,) * missing_prefix_ndim), *prefix_shape)
    if _is_broadcastable(trailing_aligned_prefix, feature_prefix_shape):
        return angles.reshape(*trailing_aligned_prefix, *token_and_pair_shape)

    raise ValueError(
        "positions 的前缀维度无法与 features 的前缀维度广播，"
        f"实际分别为 {prefix_shape} 和 {feature_prefix_shape}。"
    )


def _is_broadcastable(source_shape: tuple[int, ...], target_shape: tuple[int, ...]) -> bool:
    return all(
        source_dim == 1 or source_dim == target_dim
        for source_dim, target_dim in zip(source_shape, target_shape)
    )
