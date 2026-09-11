"""单层纯 PyTorch Mamba-3：并行训练扫描与显式增量状态。

模块: model/mamba3/mamba3.py
依赖: dataclasses, typing, torch
读取配置: —（所有结构参数由第三方调用方显式传入）
对外接口:
    - Mamba3(...) -> nn.Module                         # 单层 Mamba-3 参数容器
    - Mamba3State(angle, ssm, prev_b, prev_x) -> state # 增量推理状态
    - mamba3_forward(x, layer, mask=None, state=None, return_state=False)
        -> Tensor 或 (Tensor, Mamba3State)             # 函数式单层前向
说明: 训练路径把梯形递推改写为 affine recurrence，在固定 chunk 内以前缀积/和并行
      扫描；仅按 chunk 组织内存，不按时间步写 Python 循环。step 路径使用同一递推式，
      recurrent state 保持 FP32 以减少长序列衰减和相位累计的数值误差。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.mamba3.checks.mamba3_checks import (
    check_mamba3_inputs,
    check_mamba3_forward_inputs,
    check_mamba3_step_inputs,
)


__all__ = ["Mamba3", "Mamba3State", "mamba3_forward"]


@dataclass
class Mamba3State:
    """单层增量状态；只保存下一步梯形递推必需的量。"""

    angle: torch.Tensor
    ssm: torch.Tensor
    prev_b: torch.Tensor
    prev_x: torch.Tensor


def heavy_tail_activation(x: torch.Tensor) -> torch.Tensor:
    """正值、连续且在零点可微的 data-dependent A 激活。"""
    negative = x.clamp_max(0)
    positive = x.clamp_min(0)
    return positive + torch.reciprocal(1 - negative)


class Mamba3(nn.Module):
    """单层 Mamba-3 SSM，不包含残差、堆叠或任务头。

    所有模型超参数均由调用方显式传入，便于第三方网络自行管理配置。
    输入采用 `[B,L,d_model]`，输出保持同形；`step` 使用 `[B,d_model]`。
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        expand: int,
        headdim: int,
        ngroups: int,
        rope_fraction: float,
        dt_min: float,
        dt_max: float,
        dt_init_floor: float,
        A_floor: float,
        is_mimo: bool,
        mimo_rank: int,
        is_outproj_norm: bool,
        fuse_pregate_headwise_norm: bool,
        chunk_size: int,
        norm_eps: float,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        check_mamba3_inputs(
            d_model, d_state, expand, headdim, ngroups, rope_fraction,
            dt_min, dt_max, dt_init_floor, A_floor, is_mimo, mimo_rank,
            chunk_size, norm_eps)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.expand = int(expand)
        self.headdim = int(headdim)
        self.ngroups = int(ngroups)
        self.rope_fraction = float(rope_fraction)
        self.dt_min = float(dt_min)
        self.dt_max = float(dt_max)
        self.dt_init_floor = float(dt_init_floor)
        self.A_floor = float(A_floor)
        self.is_mimo = bool(is_mimo)
        self.mimo_rank = int(mimo_rank) if self.is_mimo else 1
        self.is_outproj_norm = bool(is_outproj_norm)
        self.fuse_pregate_headwise_norm = bool(
            fuse_pregate_headwise_norm and self.is_mimo and self.is_outproj_norm)
        self.chunk_size = int(chunk_size)
        self.norm_eps = float(norm_eps)
        self.d_inner = self.expand * self.d_model
        self.nheads = self.d_inner // self.headdim
        self.num_rope_angles = int(self.d_state * self.rope_fraction) // 2
        self.rotary_dim = self.num_rope_angles * 2

        projection_size = (
            2 * self.d_inner
            + 2 * self.d_state * self.ngroups * self.mimo_rank
            + 3 * self.nheads
            + self.num_rope_angles
        )
        self.in_proj = nn.Linear(self.d_model, projection_size, bias=False, **factory_kwargs)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False, **factory_kwargs)

        dt = torch.exp(
            torch.rand(self.nheads, device=device, dtype=torch.float32)
            * (math.log(self.dt_max) - math.log(self.dt_min))
            + math.log(self.dt_min)
        )
        dt = torch.clamp(dt, min=self.dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.B_bias = nn.Parameter(torch.ones(self.nheads, self.mimo_rank, self.d_state,
                                              dtype=torch.float32, device=device))
        self.C_bias = nn.Parameter(torch.ones(self.nheads, self.mimo_rank, self.d_state,
                                              dtype=torch.float32, device=device))
        self.B_norm_weight = nn.Parameter(torch.ones(self.d_state, **factory_kwargs))
        self.C_norm_weight = nn.Parameter(torch.ones(self.d_state, **factory_kwargs))
        self.D = nn.Parameter(torch.ones(self.nheads, dtype=torch.float32, device=device))

        if self.is_mimo:
            self.mimo_x = nn.Parameter(torch.ones(self.nheads, self.mimo_rank, self.headdim,
                                                  **factory_kwargs) / self.mimo_rank)
            self.mimo_z = nn.Parameter(torch.ones(self.nheads, self.mimo_rank, self.headdim,
                                                  **factory_kwargs))
            self.mimo_o = nn.Parameter(torch.ones(self.nheads, self.mimo_rank, self.headdim,
                                                  **factory_kwargs) / self.mimo_rank)
        else:
            self.register_parameter("mimo_x", None)
            self.register_parameter("mimo_z", None)
            self.register_parameter("mimo_o", None)
        if self.is_outproj_norm:
            self.out_norm_weight = nn.Parameter(torch.ones(self.d_inner, **factory_kwargs))
        else:
            self.register_parameter("out_norm_weight", None)

        self._projection_sizes = (
            self.d_inner,
            self.d_inner,
            self.d_state * self.ngroups * self.mimo_rank,
            self.d_state * self.ngroups * self.mimo_rank,
            self.nheads,
            self.nheads,
            self.nheads,
            self.num_rope_angles,
        )

    def allocate_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Mamba3State:
        """分配零初始化增量状态。"""
        device = device or self.in_proj.weight.device
        dtype = dtype or self.in_proj.weight.dtype
        return Mamba3State(
            angle=torch.zeros(batch_size, self.nheads, self.num_rope_angles,
                              device=device, dtype=torch.float32),
            ssm=torch.zeros(batch_size, self.nheads, self.headdim, self.d_state,
                            device=device, dtype=torch.float32),
            prev_b=torch.zeros(batch_size, self.mimo_rank, self.nheads, self.d_state,
                               device=device, dtype=dtype),
            prev_x=torch.zeros(batch_size, self.nheads, self.headdim,
                               device=device, dtype=dtype),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        state: Optional[Mamba3State] = None,
        return_state: bool = False,
    ):
        """对批量序列执行并行 Mamba-3 扫描。"""
        check_mamba3_forward_inputs(x, mask, state, self)
        batch_size, length, _ = x.shape
        state = state or self.allocate_state(batch_size, x.device, x.dtype)
        valid = mask if mask is not None else torch.ones(
            batch_size, length, device=x.device, dtype=torch.bool)
        result, final_state = self._scan(x, valid, state)
        return (result, final_state) if return_state else result

    def step(self, x: torch.Tensor, state: Mamba3State):
        """执行一个 token 的递推，返回输出和下一状态。"""
        check_mamba3_step_inputs(x, state, self)
        valid = torch.ones(x.shape[0], 1, device=x.device, dtype=torch.bool)
        output, next_state = self._scan(x.unsqueeze(1), valid, state)
        return output[:, 0], next_state

    def _project(self, x: torch.Tensor):
        z, value, b_raw, c_raw, raw_dt, raw_a, raw_trap, angle = torch.split(
            self.in_proj(x), self._projection_sizes, dim=-1)
        shape = x.shape[:-1]
        z = z.reshape(*shape, self.nheads, self.headdim)
        value = value.reshape(*shape, self.nheads, self.headdim)
        b_raw = b_raw.reshape(*shape, self.mimo_rank, self.ngroups, self.d_state)
        c_raw = c_raw.reshape(*shape, self.mimo_rank, self.ngroups, self.d_state)
        b = self._expand_groups(self._rms_norm(b_raw, self.B_norm_weight), self.B_bias)
        c = self._expand_groups(self._rms_norm(c_raw, self.C_norm_weight), self.C_bias)
        dt = F.softplus(raw_dt.float() + self.dt_bias.float())
        a = -heavy_tail_activation(raw_a.float()).clamp_min(self.A_floor)
        trap = torch.sigmoid(raw_trap.float())
        angle = angle.float().unsqueeze(-2).expand(-1, -1, self.nheads, -1)
        return z, value, b, c, dt, a, trap, angle

    def _expand_groups(self, value: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        repeat = self.nheads // self.ngroups
        if repeat == 1:
            expanded = value
        else:
            expanded = value.unsqueeze(3).expand(
                *value.shape[:3], self.ngroups, repeat, self.d_state
            ).reshape(*value.shape[:3], self.nheads, self.d_state)
        return expanded + bias.permute(1, 0, 2).unsqueeze(0).unsqueeze(0).to(expanded.dtype)

    def _rms_norm(self, value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        dtype = value.dtype
        normalized = value.float() * torch.rsqrt(
            value.float().pow(2).mean(dim=-1, keepdim=True) + self.norm_eps)
        return (normalized * weight.float()).to(dtype)

    def _scan(self, x: torch.Tensor, valid: torch.Tensor, state: Mamba3State):
        batch_size, length = int(x.shape[0]), int(x.shape[1])
        outputs = []
        h = state.ssm.float()
        previous_b = state.prev_b
        previous_x = state.prev_x
        phase_start = state.angle
        for start in range(0, length, self.chunk_size):
            end = min(start + self.chunk_size, length)
            z, value, b, c, dt, a, trap, angle = self._project(x[:, start:end])
            chunk_valid = valid[:, start:end]
            alpha = torch.exp(a * dt).float()
            beta = (1.0 - trap) * dt * alpha
            gamma = trap * dt
            valid_f = chunk_valid.unsqueeze(-1).to(alpha.dtype)
            alpha = torch.where(chunk_valid.unsqueeze(-1), alpha, torch.ones_like(alpha))
            beta = beta * valid_f
            gamma = gamma * valid_f
            phase_delta = angle * dt.unsqueeze(-1) * chunk_valid.unsqueeze(-1).unsqueeze(-1).to(angle.dtype)
            phase = phase_start.unsqueeze(1) + torch.cumsum(phase_delta, dim=1)
            b = self._rotate(b, phase)
            c = self._rotate(c, phase)
            current_x = value
            previous_x_seq = torch.cat((previous_x.unsqueeze(1), value[:, :-1]), dim=1)
            previous_b_seq = torch.cat((previous_b.unsqueeze(1), b[:, :-1]), dim=1)
            current_source = self._outer_sum(current_x, b)
            previous_source = self._outer_sum(previous_x_seq, previous_b_seq)
            source = beta.unsqueeze(-1).unsqueeze(-1) * previous_source
            source = source + gamma.unsqueeze(-1).unsqueeze(-1) * current_source
            states, h = _parallel_affine_scan(alpha, source, h)
            chunk = self._decode(states, c, current_x, z)
            chunk = chunk * chunk_valid.unsqueeze(-1).to(chunk.dtype)
            outputs.append(chunk)
            last_valid = chunk_valid[:, -1].view(batch_size, 1, 1, 1)
            previous_b = torch.where(last_valid, b[:, -1], previous_b)
            previous_x = torch.where(
                chunk_valid[:, -1].view(batch_size, 1, 1), value[:, -1], previous_x)
            phase_start = phase[:, -1]
        result = torch.cat(outputs, dim=1)
        final_state = Mamba3State(phase_start, h, previous_b, previous_x)
        return result, final_state

    def _outer_sum(self, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # x: [B,L,H,P], b: [B,L,R,H,N]；rank 分支共享一个 SSM state。
        if self.is_mimo:
            x_rank = torch.einsum("blhp,hrp->blrhp", x, self.mimo_x.to(x.dtype))
        else:
            x_rank = x.unsqueeze(2)
        return torch.einsum("blrhp,blrhn->blrhpn", x_rank, b).sum(dim=2).float()

    def _decode(self, states: torch.Tensor, c: torch.Tensor,
                x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        y_rank = torch.einsum("blrhn,blhpn->blrhp", c.float(), states)
        if self.is_mimo:
            x_rank = torch.einsum("blhp,hrp->blrhp", x, self.mimo_x.to(x.dtype))
            z_rank = torch.einsum("blhp,hrp->blrhp", z, self.mimo_z.to(z.dtype))
            y_rank = y_rank + self.D.to(x.dtype).view(1, 1, 1, self.nheads, 1) * x_rank
        else:
            x_rank = x.unsqueeze(2)
            z_rank = z.unsqueeze(2)
            y_rank = y_rank + self.D.to(x.dtype).view(1, 1, 1, self.nheads, 1) * x_rank
        if self.is_outproj_norm:
            if self.fuse_pregate_headwise_norm:
                normed = y_rank.float() * torch.rsqrt(
                    y_rank.float().pow(2).mean(dim=-1, keepdim=True) + self.norm_eps)
                y_rank = normed * self.out_norm_weight.float().view(1, 1, 1, self.nheads, self.headdim)
            else:
                shape = y_rank.shape
                y_rank = self._rms_norm(y_rank.reshape(*shape[:-2], self.d_inner),
                                        self.out_norm_weight).reshape(shape)
        y_rank = y_rank * F.silu(z_rank)
        if self.is_mimo:
            y = torch.einsum("blrhp,hrp->blhp", y_rank, self.mimo_o.to(y_rank.dtype))
        else:
            y = y_rank[:, :, 0]
        return self.out_proj(y.reshape(y.shape[0], y.shape[1], self.d_inner).to(self.out_proj.weight.dtype))

    def _rotate(self, value: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        if self.rotary_dim == 0:
            return value
        first = value[..., :self.rotary_dim:2]
        second = value[..., 1:self.rotary_dim:2]
        cosine = torch.cos(phase).unsqueeze(2)
        sine = torch.sin(phase).unsqueeze(2)
        rotated_first = first * cosine - second * sine
        rotated_second = first * sine + second * cosine
        rotated = torch.stack((rotated_first, rotated_second), dim=-1).flatten(-2)
        return torch.cat((rotated, value[..., self.rotary_dim:]), dim=-1)


def _parallel_affine_scan(alpha: torch.Tensor, source: torch.Tensor, initial: torch.Tensor):
    """按 chunk 内前缀积/和并行求解 h_t=alpha_t*h_(t-1)+source_t。"""
    prefix = torch.cumprod(alpha.float(), dim=1)
    scaled = source / prefix.unsqueeze(-1).unsqueeze(-1)
    states = prefix.unsqueeze(-1).unsqueeze(-1) * (
        initial.unsqueeze(1) + torch.cumsum(scaled, dim=1))
    return states, states[:, -1]


def mamba3_forward(
    x: torch.Tensor,
    layer: Mamba3,
    mask: Optional[torch.Tensor] = None,
    state: Optional[Mamba3State] = None,
    return_state: bool = False,
):
    """第三方调用的函数式单层 Mamba-3 前向入口。"""
    return layer(x, mask=mask, state=state, return_state=return_state)
