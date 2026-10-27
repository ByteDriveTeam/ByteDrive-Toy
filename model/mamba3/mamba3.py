"""单层纯 PyTorch Mamba-3：并行训练扫描与显式增量状态。

模块: model/mamba3/mamba3.py
依赖: dataclasses, math, typing, torch
读取配置: —（所有结构参数由第三方调用方显式传入）
对外接口:
    - Mamba3(...) -> nn.Module                         # 单层 Mamba-3 参数容器
    - Mamba3.forward_checkpointed(...) -> Tensor 或 (Tensor, Mamba3State)
    - Mamba3.diagnostics(x, mask=None, state=None) -> dict
    - Mamba3State(angle, ssm, prev_b, prev_x) -> state # 可克隆、截断梯度和按 batch 重置
    - mamba3_forward(x, layer, mask=None, state=None, return_state=False, reset=None,
                     checkpoint_chunks=False)
        -> Tensor 或 (Tensor, Mamba3State)             # 函数式单层前向
说明: 输入/输出投影支持 AMP，其余参数及状态计算保持 FP32；B/x 缓存使用指定或模型
      投影 dtype。扫描与 step 共用分段仿射递推，累计相位按 2π 取模；mask 暂停更新，
      reset 在 token 前清零。返回状态具有独立紧凑存储，默认保留跨调用梯度。
      MIMO 的 D 初值按 rank 补偿；重计算通过正常模块调用入口选择。
      fuse_pregate_headwise_norm 仅为兼容参数，纯 PyTorch 路径没有独立融合 kernel。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from model.mamba3.checks.mamba3_checks import (
    check_mamba3_inputs,
    check_mamba3_forward_inputs,
    check_mamba3_step_inputs,
    check_mamba3_reset_inputs,
)


__all__ = ["Mamba3", "Mamba3State", "mamba3_forward"]


@dataclass
class Mamba3State:
    """单层增量状态；只保存下一步梯形递推必需的量。"""

    angle: torch.Tensor
    ssm: torch.Tensor
    prev_b: torch.Tensor
    prev_x: torch.Tensor

    def _map(self, fn):
        return Mamba3State(*(fn(t) for t in (self.angle, self.ssm, self.prev_b, self.prev_x)))

    def clone(self) -> Mamba3State:
        """复制为独立紧凑存储，保留梯度连接。"""
        return self._map(torch.clone)

    def detach(self) -> Mamba3State:
        """截断跨调用梯度并复制存储，供截断反向传播或推理缓存使用。"""
        return self._map(lambda t: t.detach().clone())

    def reset(self, mask: torch.Tensor) -> Mamba3State:
        """按 [B] bool 掩码清零 batch 状态，保留其他 batch 的梯度。"""
        check_mamba3_reset_inputs(mask, self.angle.shape[:1], self.angle.device)
        return self._map(lambda t: torch.where(
            mask.reshape(mask.shape[0], *((1,) * (t.ndim - 1))), torch.zeros_like(t), t))


def heavy_tail_activation(x: torch.Tensor) -> torch.Tensor:
    """正值、连续且在零点可微的 data-dependent A 激活。"""
    # 零点只选择正分支，避免两个 clamp 同时贡献梯度；负分支分母始终安全。
    return torch.where(x >= 0, 1 + x, torch.reciprocal(1 - x.clamp_max(0)))


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
            chunk_size, norm_eps, is_outproj_norm, fuse_pregate_headwise_norm)
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
        self.B_norm_weight = nn.Parameter(torch.ones(self.d_state, dtype=torch.float32, device=device))
        self.C_norm_weight = nn.Parameter(torch.ones(self.d_state, dtype=torch.float32, device=device))
        # mimo_x 与 mimo_o 各含 1/rank；D 乘 rank 恢复初始化时的有效直通增益。
        self.D = nn.Parameter(torch.full((self.nheads,), float(self.mimo_rank),
                                        dtype=torch.float32, device=device))
        self.dt_bias._no_weight_decay = True
        self.D._no_weight_decay = True

        if self.is_mimo:
            self.mimo_x = nn.Parameter(torch.ones(self.nheads, self.mimo_rank, self.headdim,
                                                  dtype=torch.float32, device=device) / self.mimo_rank)
            self.mimo_z = nn.Parameter(torch.ones(self.nheads, self.mimo_rank, self.headdim,
                                                  dtype=torch.float32, device=device))
            self.mimo_o = nn.Parameter(torch.ones(self.nheads, self.mimo_rank, self.headdim,
                                                  dtype=torch.float32, device=device) / self.mimo_rank)
        else:
            self.register_parameter("mimo_x", None)
            self.register_parameter("mimo_z", None)
            self.register_parameter("mimo_o", None)
        if self.is_outproj_norm:
            self.out_norm_weight = nn.Parameter(torch.ones(self.d_inner, dtype=torch.float32, device=device))
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

    def _apply(self, fn, recurse=True):
        # 先用空张量确定目标设备，避免核心参数先量化再转回 FP32 而永久损失精度。
        protected = {id(t) for p in self.parameters(recurse=False)
                     for t in (p, p.grad) if t is not None}

        def convert(tensor):
            if id(tensor) not in protected or tensor.is_meta:
                result = fn(tensor)
                return result.float() if id(tensor) in protected else result
            target = fn(torch.empty(0, device=tensor.device, dtype=tensor.dtype))
            return tensor.to(device=target.device, dtype=torch.float32)

        return super()._apply(convert, recurse=recurse)

    def allocate_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Mamba3State:
        """分配零状态；angle/ssm 固定 FP32，prev_b/prev_x 使用指定或模型 dtype。"""
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
        reset: Optional[torch.Tensor] = None,
        *, checkpoint_chunks: bool = False,
    ):
        """对批量序列执行并行 Mamba-3 扫描。"""
        return self._forward(x, mask, state, return_state, reset, checkpoint_chunks)

    def forward_checkpointed(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
        state: Optional[Mamba3State] = None, return_state: bool = False,
        reset: Optional[torch.Tensor] = None,
    ):
        """重计算的兼容入口；经 __call__ 调用以执行本模块的 forward hooks。"""
        return self(x, mask=mask, state=state, return_state=return_state,
                    reset=reset, checkpoint_chunks=True)

    def _checkpoint_chunk(self, x, valid, state, reset):
        if not torch.is_grad_enabled():
            return self._scan_chunk(x, valid, state, reset)
        return checkpoint(self._scan_chunk, x, valid, state, reset, use_reentrant=False)

    def _forward(self, x, mask, state, return_state, reset, checkpoint_chunks):
        check_mamba3_forward_inputs(x, mask, state, self, reset, checkpoint_chunks)
        batch_size, length, _ = x.shape
        state = state or self.allocate_state(batch_size, x.device)
        valid = mask if mask is not None else torch.ones(
            batch_size, length, device=x.device, dtype=torch.bool)
        resets = reset if reset is not None else torch.zeros_like(valid)
        if length == 0:
            result, final_state = x.new_empty(batch_size, 0, self.d_model), state.clone()
        else:
            scan_chunk = self._checkpoint_chunk if checkpoint_chunks else self._scan_chunk
            result, final_state = self._scan(x, valid, state, resets, scan_chunk)
        return (result, final_state) if return_state else result

    def step(self, x: torch.Tensor, state: Mamba3State,
             mask: Optional[torch.Tensor] = None, reset: Optional[torch.Tensor] = None):
        """递推一个 token；[B] mask 暂停更新，[B] reset 在当前 token 前清零。"""
        check_mamba3_step_inputs(x, state, self, mask, reset)
        valid = mask if mask is not None else torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        resets = reset if reset is not None else torch.zeros_like(valid)
        output, next_state = self._scan_chunk(
            x.unsqueeze(1), valid.unsqueeze(1), state, resets.unsqueeze(1))
        return output[:, 0], next_state

    def _project(self, x: torch.Tensor):
        projected = self.in_proj(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self._project_parts(projected, x.shape[:-1])

    def _project_parts(self, projected: torch.Tensor, shape):
        z, value, b_raw, c_raw, raw_dt, raw_a, raw_trap, angle = torch.split(
            projected, self._projection_sizes, dim=-1)
        z = z.reshape(*shape, self.nheads, self.headdim)
        value = value.reshape(*shape, self.nheads, self.headdim)
        b_raw = b_raw.reshape(*shape, self.mimo_rank, self.ngroups, self.d_state)
        c_raw = c_raw.reshape(*shape, self.mimo_rank, self.ngroups, self.d_state)
        b = self._expand_groups(self._rms_norm(b_raw.float(), self.B_norm_weight), self.B_bias)
        c = self._expand_groups(self._rms_norm(c_raw.float(), self.C_norm_weight), self.C_bias)
        dt = F.softplus(raw_dt.float() + self.dt_bias.float())
        a = -heavy_tail_activation(raw_a.float()).clamp_min(self.A_floor)
        trap = torch.sigmoid(raw_trap.float())
        angle = (math.pi * torch.tanh(angle.float())).unsqueeze(-2).expand(
            -1, -1, self.nheads, -1)
        return z, value, b, c, dt, a, trap, angle

    def _expand_groups(self, value: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        repeat = self.nheads // self.ngroups
        expanded = value.repeat_interleave(repeat, dim=-2) if repeat != 1 else value
        return expanded.float() + bias.float().permute(1, 0, 2).unsqueeze(0).unsqueeze(0)

    def _rms_norm(self, value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        dtype = value.dtype
        value = value.float()
        # 缩放在代数上抵消，无需对最大值反传；epsilon 同步缩放以保留小信号行为。
        scale = value.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1)
        scaled = value / scale
        normalized = scaled * torch.rsqrt(
            scaled.square().mean(dim=-1, keepdim=True) + (math.sqrt(self.norm_eps) / scale).square())
        return (normalized * weight.float()).to(dtype)

    def _scan(self, x, valid, state, reset, scan_chunk):
        outputs = []
        for start in range(0, x.shape[1], self.chunk_size):
            end = start + self.chunk_size
            chunk, state = scan_chunk(x[:, start:end], valid[:, start:end], state, reset[:, start:end])
            outputs.append(chunk)
        return torch.cat(outputs, dim=1), state

    def _scan_chunk(self, x, valid, state, reset):
        # 无效输入在投影前清零，避免 NaN/Inf 进入线性层与反向传播。
        projected = self._project(torch.where(valid.unsqueeze(-1), x, torch.zeros_like(x)))
        with torch.autocast(device_type=x.device.type, enabled=False):
            y, next_state = self._scan_projected(projected, valid, state, reset)
        output = self.out_proj(y.flatten(-2).to(self.out_proj.weight.dtype))
        return torch.where(valid.unsqueeze(-1), output, torch.zeros_like(output)), next_state

    def _scan_projected(self, projected, valid, state, reset):
        z, value, b, c, dt, a, trap, angle = projected
        batch, length = valid.shape
        positions = torch.arange(1, length + 1, device=valid.device).expand(batch, -1)
        zero = positions.new_zeros(batch, 1)
        segments = reset.long().cumsum(1)
        alpha = torch.exp(a * dt)
        beta = torch.where(valid.unsqueeze(-1), (1 - trap) * dt * alpha, 0)
        gamma = torch.where(valid.unsqueeze(-1), trap * dt, 0)
        alpha = torch.where(reset.unsqueeze(-1), 0,
                            torch.where(valid.unsqueeze(-1), alpha, 1))
        phase = self._segment_phase(angle * dt.unsqueeze(-1), valid, reset, state.angle, positions)
        b, c = self._rotate(b, phase), self._rotate(c, phase)
        valid_positions = torch.where(valid, positions, 0)
        predecessor = torch.cummax(torch.cat((zero, valid_positions[:, :-1]), 1), 1).values
        segment_history = torch.cat((zero, segments), 1)
        same_segment = segment_history.gather(1, predecessor) == segments
        x_history = torch.cat((state.prev_x.unsqueeze(1), value.to(state.prev_x.dtype)), 1)
        b_history = torch.cat((state.prev_b.unsqueeze(1), b.to(state.prev_b.dtype)), 1)
        prev_x = self._gather_history(x_history, predecessor, same_segment)
        prev_b = self._gather_history(b_history, predecessor, same_segment)
        source = (beta[..., None, None] * self._outer_sum(prev_x, prev_b)
                  + gamma[..., None, None] * self._outer_sum(value, b))
        states, h = _parallel_affine_scan(alpha, source, state.ssm)
        y = self._decode_core(states, c, value, z)
        latest = valid_positions.cummax(1).values[:, -1:]
        same_final_segment = segment_history.gather(1, latest) == segments[:, -1:]
        final_x = self._gather_history(x_history, latest, same_final_segment)[:, 0]
        final_b = self._gather_history(b_history, latest, same_final_segment)[:, 0]
        return y, Mamba3State(phase[:, -1], h, final_b, final_x).clone()

    @staticmethod
    def _gather_history(history, indices, keep):
        shape = (*indices.shape, *((1,) * (history.ndim - 2)))
        gathered = history.gather(1, indices.reshape(shape).expand(*indices.shape, *history.shape[2:]))
        return torch.where(keep.reshape(shape), gathered, torch.zeros_like(gathered))

    @staticmethod
    def _segment_phase(delta, valid, reset, initial, positions):
        cumulative = torch.where(valid[..., None, None], delta, 0).cumsum(1)
        last_reset = torch.where(reset, positions, 0).cummax(1).values
        history = torch.cat((torch.zeros_like(initial).unsqueeze(1), cumulative), 1)
        before_reset = history.gather(
            1, (last_reset - 1).clamp_min(0)[..., None, None].expand_as(cumulative))
        base = torch.where(last_reset[..., None, None] == 0, initial.unsqueeze(1), -before_reset)
        phase = torch.remainder(base + cumulative, 2 * math.pi)
        untouched = (valid.long().cumsum(1) == 0) & (last_reset == 0)
        return torch.where(untouched[..., None, None], initial.unsqueeze(1), phase)

    def _outer_sum(self, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # 在收缩时归约 rank，避免构造 [B,L,R,H,P,N] 的外积中间张量。
        with torch.autocast(device_type=x.device.type, enabled=False):
            mixed_b = (torch.einsum("hrp,blrhn->blhpn", self.mimo_x.float(), b.float())
                       if self.is_mimo else b[:, :, 0].float().unsqueeze(-2))
            return x.float().unsqueeze(-1) * mixed_b

    def _decode(self, states: torch.Tensor, c: torch.Tensor,
                x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            y = self._decode_core(states, c, x, z)
        return self.out_proj(y.flatten(-2).to(self.out_proj.weight.dtype))

    def _decode_core(self, states, c, x, z):
        y_rank = torch.einsum("blrhn,blhpn->blrhp", c.float(), states.float())
        x_rank, z_rank = x.float().unsqueeze(2), z.float().unsqueeze(2)
        if self.is_mimo:
            x_rank = x_rank * self.mimo_x.permute(1, 0, 2).float()
            z_rank = z_rank * self.mimo_z.permute(1, 0, 2).float()
        y_rank = y_rank + self.D.float().view(1, 1, 1, self.nheads, 1) * x_rank
        if self.is_outproj_norm:
            # 纯 PyTorch 的两条路径共享按 head 归一化，保持官方 group_size=headdim 的定义。
            y_rank = self._rms_norm(
                y_rank, self.out_norm_weight.view(self.nheads, self.headdim))
        y_rank = y_rank * F.silu(z_rank)
        if self.is_mimo:
            y = torch.einsum("blrhp,hrp->blhp", y_rank, self.mimo_o.float())
        else:
            y = y_rank[:, :, 0]
        return y

    @torch.no_grad()
    def diagnostics(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                    state: Optional[Mamba3State] = None) -> dict:
        """显式采样 dt、衰减、相位增量、tanh 饱和与状态范数；不修改参数或缓存。"""
        check_mamba3_forward_inputs(x, mask, state, self)
        valid = mask if mask is not None else torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        result = {"valid_tokens": int(valid.sum().item())}
        if result["valid_tokens"]:
            *_, dt, a, _, rate = self._project(torch.where(valid[..., None], x, 0))
            samples = {"dt": dt[valid], "abs_adt": (a * dt).abs()[valid],
                       "abs_phase_delta": (rate * dt[..., None]).abs()[valid]}
            for name, sample in samples.items():
                result.update({name + "_min": sample.min().item(), name + "_mean": sample.mean().item(),
                               name + "_p95": sample.flatten().quantile(0.95).item(),
                               name + "_max": sample.max().item()})
            tanh = rate[valid] / math.pi
            result["tanh_saturated_fraction"] = (tanh.abs() == 1).float().mean().item()
            result["tanh_derivative_mean"] = (1 - tanh.square()).mean().item()
        if state is not None:
            result["ssm_rms"] = state.ssm.square().mean().sqrt().item()
            result["ssm_abs_max"] = state.ssm.abs().max().item()
        return result

    def _rotate(self, value: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        if self.rotary_dim == 0:
            return value
        if self.is_mimo:
            # 官方 MIMO 按完整 d_state 分半；部分旋转时两半各自的尾部保持原值。
            half = self.d_state // 2
            first = value[..., :self.num_rope_angles]
            second = value[..., half:half + self.num_rope_angles]
        else:
            first = value[..., :self.rotary_dim:2]
            second = value[..., 1:self.rotary_dim:2]
        cosine = torch.cos(phase).unsqueeze(2)
        sine = torch.sin(phase).unsqueeze(2)
        rotated_first = first * cosine - second * sine
        rotated_second = first * sine + second * cosine
        if self.is_mimo:
            rotated = torch.cat((rotated_first, value[..., self.num_rope_angles:half],
                                 rotated_second, value[..., half + self.num_rope_angles:]), dim=-1)
        else:
            rotated = torch.stack((rotated_first, rotated_second), dim=-1).flatten(-2)
            rotated = torch.cat((rotated, value[..., self.rotary_dim:]), dim=-1)
        # 相位用 FP32 计算，旋转结果按官方 Q/K 输出缓冲区的 dtype 保存。
        return rotated.to(value.dtype)


def _parallel_affine_scan(alpha: torch.Tensor, source: torch.Tensor, initial: torch.Tensor):
    """不使用除法的仿射递推包含式并行前缀扫描。"""
    coefficient = alpha.float()
    offset = source.float()
    stride = 1
    while stride < coefficient.shape[1]:
        right = coefficient[:, stride:]
        scale = right.unsqueeze(-1).unsqueeze(-1)
        # reset 的零系数必须真正截断旧段，避免 0*NaN/Inf 使新段继续污染。
        prior = torch.where(scale == 0, 0, offset[:, :-stride])
        offset = torch.cat((offset[:, :stride],
                            offset[:, stride:] + scale * prior), dim=1)
        coefficient = torch.cat((coefficient[:, :stride],
                                 right * torch.where(right == 0, 0, coefficient[:, :-stride])), dim=1)
        stride *= 2
    scale = coefficient.unsqueeze(-1).unsqueeze(-1)
    states = scale * torch.where(scale == 0, 0, initial.unsqueeze(1)) + offset
    return states, states[:, -1]


def mamba3_forward(
    x: torch.Tensor,
    layer: Mamba3,
    mask: Optional[torch.Tensor] = None,
    state: Optional[Mamba3State] = None,
    return_state: bool = False,
    reset: Optional[torch.Tensor] = None,
    *, checkpoint_chunks: bool = False,
):
    """第三方调用的函数式单层 Mamba-3 前向入口。"""
    return layer(x, mask=mask, state=state, return_state=return_state, reset=reset,
                 checkpoint_chunks=checkpoint_chunks)
