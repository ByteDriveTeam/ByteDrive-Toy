"""相机 Patch 中心及四角的深度射线坐标生成器。

模块: model/frustum_encoding/frustum_encoding.py
依赖: torch, model.frustum_encoding.checks.frustum_encoding_checks
读取配置: —（射线采样参数由调用方传入，来源 model.driving.frustum）
对外接口:
    - FrustumEncoding(...) -> nn.Module
        ego_frustum_coords(gh, gw, intrinsics, extrinsics) -> Tensor
        forward(patch_features, intrinsics, extrinsics) -> Tensor
说明: 本模块只生成 ego 系几何坐标；查询与被查询方的两个射线 MLP 均由驾驶 Transformer 持有。
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from model.frustum_encoding.checks.frustum_encoding_checks import (
    check_frustum_args,
    check_frustum_geometry,
    check_frustum_inputs,
)


__all__ = ["FrustumEncoding"]

# 每 patch 采样像素：中心 + 四角，偏移以 patch 为单位的 (列 c_frac, 行 r_frac)
_PIXEL_OFFSETS = ((0.5, 0.5), (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))


class FrustumEncoding(nn.Module):
    """生成每个 Patch 的视锥候选 3D 坐标，不改写视觉 Token。

    Shape:
        patch_features: `[B, C, gh, gw]`（仅用于取 gh/gw/device）
        intrinsics: `[B, 4]`（fx, fy, cx, cy）
        extrinsics: `[B, 6]`（x, y, z, roll, pitch, yaw；相机在 ego 系位姿）
        输出: `[B, gh*gw, 5, N, 3]`
    """

    def __init__(self, out_dim: int, patch_size: int, depth_min_m: float, depth_max_m: float,
                 step_near_m: float, step_far_m: float, coord_symlog_scale: float,
                 mlp_hidden: int) -> None:
        super().__init__()
        check_frustum_args(out_dim, patch_size, mlp_hidden, coord_symlog_scale)
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.coord_symlog_scale = coord_symlog_scale
        # 深度采样为一维常量，随模型保存/搬设备
        self.register_buffer("depths", _build_depth_samples(
            depth_min_m, depth_max_m, step_near_m, step_far_m))
        self._pixel_cache: Dict[tuple, torch.Tensor] = {}  # (gh,gw)->[gh*gw,5,2] 像素 (u,v)

    def forward(self, patch_features: torch.Tensor, intrinsics: torch.Tensor,
                extrinsics: torch.Tensor) -> torch.Tensor:
        """返回纯几何射线；位置 MLP 由调用方的 Q/K 两个共享编码头持有。"""
        check_frustum_inputs(patch_features, intrinsics, extrinsics, self.patch_size)
        gh, gw = int(patch_features.shape[2]), int(patch_features.shape[3])
        return self._ego_coords(gh, gw, intrinsics.float(), extrinsics.float())

    def ego_frustum_coords(self, gh: int, gw: int, intrinsics: torch.Tensor,
                           extrinsics: torch.Tensor) -> torch.Tensor:
        """对外暴露 ego 系候选坐标 `[B, gh*gw, 5, N, 3]`（几何 FP32），供 BEV 侧几何先验使用。"""
        check_frustum_geometry(gh, gw, intrinsics, extrinsics)
        return self._ego_coords(gh, gw, intrinsics.float(), extrinsics.float())

    def _ego_coords(self, gh: int, gw: int, intrinsics: torch.Tensor,
                    extrinsics: torch.Tensor) -> torch.Tensor:
        """反投影每 patch 5 像素 × N 深度到 ego 系 3D 坐标：`[B, gh*gw, 5, N, 3]`。"""
        pixels = self._pixels(gh, gw, intrinsics.device)          # [P,5,2] (u,v)
        depths = self.depths.to(intrinsics.device)                # [N]
        fx, fy, cx, cy = (intrinsics[:, i][:, None, None] for i in range(4))  # 各 [B,1,1]
        u = pixels[..., 0].unsqueeze(0)                           # [1,P,5]
        v = pixels[..., 1].unsqueeze(0)
        # 归一化视线方向（与深度无关）：像平面系 (x右,y下,z前) 的 X/Z, Y/Z
        dir_x = (u - cx) / fx                                     # [B,P,5]
        dir_y = (v - cy) / fy
        d = depths.view(1, 1, 1, -1)                              # [1,1,1,N]
        img_x = dir_x.unsqueeze(-1) * d                           # [B,P,5,N]
        img_y = dir_y.unsqueeze(-1) * d
        img_z = d.expand_as(img_x)
        # 像平面系(x右,y下,z前) → 传感器系(x前,y右,z上)：前=z, 右=x, 上=-y
        sensor = torch.stack((img_z, img_x, -img_y), dim=-1)      # [B,P,5,N,3]
        rot, trans = _carla_rotation(extrinsics)                  # [B,3,3],[B,3]
        ego = torch.einsum("bij,bpqnj->bpqni", rot, sensor) + trans[:, None, None, None, :]
        return ego

    def _pixels(self, gh: int, gw: int, device: torch.device) -> torch.Tensor:
        """惰性构建并缓存 (gh,gw) 的每 patch 5 像素 (u,v) 坐标 `[gh*gw, 5, 2]`（行主序 r*gw+c）。"""
        key = (gh, gw)
        cached = self._pixel_cache.get(key)
        if cached is None or cached.device != device:
            cached = _build_pixels(gh, gw, self.patch_size, device)
            self._pixel_cache[key] = cached
        return cached

def _build_pixels(gh: int, gw: int, patch_size: int, device: torch.device) -> torch.Tensor:
    """每 patch 中心+四角像素 (u,v)：`[gh*gw, 5, 2]`，行主序与 patch 特征 flatten 对齐。"""
    rows = torch.arange(gh, dtype=torch.float32, device=device)
    cols = torch.arange(gw, dtype=torch.float32, device=device)
    grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")   # [gh,gw]
    offsets = torch.tensor(_PIXEL_OFFSETS, dtype=torch.float32, device=device)  # [5,(c_frac,r_frac)]
    u = (grid_c[..., None] + offsets[:, 0]) * patch_size         # [gh,gw,5]
    v = (grid_r[..., None] + offsets[:, 1]) * patch_size
    return torch.stack((u, v), dim=-1).reshape(gh * gw, len(_PIXEL_OFFSETS), 2)


def _build_depth_samples(depth_min: float, depth_max: float,
                         step_near: float, step_far: float) -> torch.Tensor:
    """近密远疏深度采样：步长随深度从 step_near 线性增到 step_far，累进直到超过 depth_max。

    步长随深度而非索引线性变化，故用累进 while（有依赖、不可向量化）——一次性构建常量 buffer（§9 第 4 档）。
    """
    span = max(depth_max - depth_min, 1e-6)
    samples = []
    d = depth_min
    while d <= depth_max:
        samples.append(d)
        frac = min(max((d - depth_min) / span, 0.0), 1.0)
        d += step_near + (step_far - step_near) * frac
    return torch.tensor(samples, dtype=torch.float32)


def _carla_rotation(extrinsics: torch.Tensor) -> tuple:
    """由 [B,6]=(x,y,z,roll,pitch,yaw)(米/度) 构造 CARLA 左手系旋转 [B,3,3] 与平移 [B,3]（复刻 vis geometry）。"""
    trans = extrinsics[:, :3]
    rad = torch.deg2rad(extrinsics[:, 3:])
    cr, sr = torch.cos(rad[:, 0]), torch.sin(rad[:, 0])
    cp, sp = torch.cos(rad[:, 1]), torch.sin(rad[:, 1])
    cy, sy = torch.cos(rad[:, 2]), torch.sin(rad[:, 2])
    row0 = torch.stack((cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr), dim=-1)
    row1 = torch.stack((sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr), dim=-1)
    row2 = torch.stack((sp, -cp * sr, cp * cr), dim=-1)
    return torch.stack((row0, row1, row2), dim=1), trans
