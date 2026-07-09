"""监督目标编码：Symlog 物理量、深度范围掩码、光流→图像平面速度的纯函数。

模块: data/target_encoding.py
依赖: torch, data.target_encoding_checks
读取配置: —（缩放/量程/内参/时间步等均由调用方以参数传入，来源为 config.model.physics 与场景内参）
对外接口:
    - symlog(x) / inv_symlog(y) -> Tensor                     # sign(x)·ln(1+|x|) 及其逆
    - physics_target(x, scale) / physics_decode(y, scale) -> Tensor   # scale·symlog 及其逆
    - depth_targets(depth_m, scale, depth_max_m) -> (target, in_range)  # 深度回归目标与范围内掩码
    - flow_velocity_targets(flow, depth_m, fx, fy, ndc_scale, dt, scale) -> Tensor  # [.,2,H,W] 速度目标
说明: 物理量在 Symlog 空间监督：目标 = scale·symlog(物理量)，推理走 physics_decode 逆变换。深度仅在
      range 内回归（超范围如天空由 in_range=0 掩掉，另以二分类监督）。光流为 CARLA 归一化图像运动，
      借当前帧深度 Z 与内参 fx/fy 反投影为图像平面度量位移，再除时间步 dt 得速度（m/s，2 通道，非全 3D
      场景流）。ndc_scale 为归一化光流→像素位移的有符号缩放 [sx, sy]，符号/量级在实现期经验校正。
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch

from data.target_encoding_checks import check_flow_inputs, check_map_2d


__all__ = [
    "symlog", "inv_symlog", "physics_target", "physics_decode",
    "depth_targets", "flow_velocity_targets",
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


def flow_velocity_targets(flow: torch.Tensor, depth_m: torch.Tensor, fx: float, fy: float,
                          ndc_scale: Sequence[float], dt: float, scale: float) -> torch.Tensor:
    """CARLA 归一化光流 → 图像平面度量速度的 symlog 目标。

    参数:
        flow:      光流图，形状 [..., H, W, 2]，最后一维为 (vx, vy)（归一化图像运动）
        depth_m:   当前帧深度（米），形状 [..., H, W]（CARLA 光流为后向，矢量落在当前帧像素上）
        fx, fy:    针孔内参焦距（像素）
        ndc_scale: 归一化光流→像素位移的有符号缩放 [sx, sy]（通常 [W/2, H/2]）
        dt:        时间步（秒），= 仿真 fixed_delta_seconds
        scale:     symlog 缩放因子
    返回:
        速度目标，形状 [..., 2, H, W]（= scale·symlog(velocity)，单位 m/s 的 symlog）
    """
    check_flow_inputs(flow, depth_m)
    # 归一化光流 → 像素位移（有符号缩放，吸收 CARLA 的 NDC 满屏约定与 y 轴朝下）
    du = flow[..., 0] * ndc_scale[0]
    dv = flow[..., 1] * ndc_scale[1]
    # 像素位移 → 当前深度处的图像平面度量位移（针孔反投影），再除 dt 得速度
    velocity_x = du * depth_m / (fx * dt)
    velocity_y = dv * depth_m / (fy * dt)
    velocity = torch.stack((velocity_x, velocity_y), dim=-3)  # [..., 2, H, W]
    return physics_target(velocity, scale)
