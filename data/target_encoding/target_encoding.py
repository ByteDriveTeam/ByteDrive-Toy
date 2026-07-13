"""监督目标编码：Symlog 物理量、深度范围掩码的纯函数。

模块: data/target_encoding/target_encoding.py
依赖: torch, data.target_encoding.checks.target_encoding_checks
读取配置: —（缩放/量程等均由调用方以参数传入，来源为 config.model.physics）
对外接口:
    - symlog(x) / inv_symlog(y) -> Tensor                     # sign(x)·ln(1+|x|) 及其逆
    - physics_target(x, scale) / physics_decode(y, scale) -> Tensor   # scale·symlog 及其逆
    - depth_targets(depth_m, scale, depth_max_m) -> (target, in_range)  # 深度回归目标与范围内掩码
说明: 物理量在 Symlog 空间监督：目标 = scale·symlog(物理量)，推理走 physics_decode 逆变换。深度仅在
      range 内回归（超范围如天空由 in_range=0 掩掉，另以二分类监督）。
"""

from __future__ import annotations

from typing import Tuple

import torch

from data.target_encoding.checks.target_encoding_checks import check_map_2d


__all__ = [
    "symlog", "inv_symlog", "physics_target", "physics_decode",
    "depth_targets",
]


def symlog(x: torch.Tensor) -> torch.Tensor:
    """对称对数压缩：sign(x)·ln(1+|x|)，压缩大动态范围又保号、在 0 附近近似线性。"""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def inv_symlog(y: torch.Tensor) -> torch.Tensor:
    """symlog 的逆：sign(y)·(exp(|y|)-1)。"""
    return torch.sign(y) * torch.expm1(torch.abs(y))


def physics_target(x: torch.Tensor, scale: float) -> torch.Tensor:
    """物理量 → 监督空间：scale·symlog(x)。"""
    return scale * symlog(x)


def physics_decode(y: torch.Tensor, scale: float) -> torch.Tensor:
    """监督空间 → 物理量：inv_symlog(y/scale)。"""
    return inv_symlog(y / scale)


def depth_targets(depth_m: torch.Tensor, scale: float,
                  depth_max_m: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """深度回归目标与范围内掩码。

    参数:
        depth_m: 深度图（米），形状 [..., H, W]
        scale:   symlog 缩放因子
        depth_max_m: 检测上限（米），< 上限记为范围内
    返回:
        (target, in_range)：target = scale·symlog(depth)（全图算，回归时按掩码取用）；
        in_range 为 float mask（1=范围内，0=超范围），既作回归掩码也作二分类标签。
    """
    check_map_2d(depth_m, "depth_m")
    in_range = (depth_m < depth_max_m).to(depth_m.dtype)
    return physics_target(depth_m, scale), in_range
