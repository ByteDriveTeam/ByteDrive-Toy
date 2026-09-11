# 本文件为 model/mamba3/mamba3.py 的校验伴随文件（规范 §7.1，免文件头）。

import torch


def check_mamba3_inputs(d_model, d_state, expand, headdim, ngroups, rope_fraction,
                        dt_min, dt_max, dt_init_floor, a_floor, is_mimo,
                        mimo_rank, chunk_size, norm_eps):
    """校验对象: Mamba3 构造参数 —— 单层投影、状态和稳定性参数必须可实现。"""
    if d_model <= 0 or d_state <= 0 or expand <= 0 or headdim <= 0:
        raise ValueError("d_model/d_state/expand/headdim 必须为正整数。")
    if d_model * expand % headdim != 0:
        raise ValueError("d_model*expand 必须被 headdim 整除。")
    if ngroups <= 0 or (d_model * expand // headdim) % ngroups != 0:
        raise ValueError("ngroups 必须为正数且整除 heads。")
    if rope_fraction not in (0.5, 1.0):
        raise ValueError("rope_fraction 仅支持 0.5 或 1.0。")
    if int(d_state * rope_fraction) < 2 or int(d_state * rope_fraction) % 2:
        raise ValueError("rope_fraction*d_state 必须产生至少一对偶数旋转维。")
    if not (0 < dt_min <= dt_max) or dt_init_floor <= 0 or a_floor <= 0:
        raise ValueError("dt_min/dt_max/dt_init_floor/A_floor 参数非法。")
    if mimo_rank <= 0 or chunk_size <= 0 or norm_eps <= 0:
        raise ValueError("mimo_rank/chunk_size/norm_eps 必须为正数。")
    if not is_mimo and mimo_rank != 1:
        raise ValueError("SISO 模式的 mimo_rank 必须为 1。")


def _check_mamba3_state(state, layer, batch_size, device):
    """校验对象: Mamba3State —— 增量状态布局须与单层结构匹配。"""
    expected = (
        (batch_size, layer.nheads, layer.num_rope_angles),
        (batch_size, layer.nheads, layer.headdim, layer.d_state),
        (batch_size, layer.mimo_rank, layer.nheads, layer.d_state),
        (batch_size, layer.nheads, layer.headdim),
    )
    actual = (state.angle.shape, state.ssm.shape, state.prev_b.shape, state.prev_x.shape)
    if actual != expected:
        raise ValueError("Mamba3State 形状非法，期望 {}，实际 {}。".format(expected, actual))
    if any(t.device != device for t in (state.angle, state.ssm, state.prev_b, state.prev_x)):
        raise ValueError("Mamba3State 必须与输入位于同一 device。")


def check_mamba3_forward_inputs(x, mask, state, layer):
    """校验对象: Mamba3.forward 的全部入参。"""
    if x.ndim != 3 or int(x.shape[-1]) != layer.d_model:
        raise ValueError("x 期望 [B,L,d_model={}], 实际 {}。".format(layer.d_model, tuple(x.shape)))
    if mask is not None and (mask.ndim != 2 or mask.shape[:2] != x.shape[:2]
                             or mask.dtype != torch.bool
                             or mask.device != x.device):
        raise ValueError("mask 期望 [B,L] bool，实际 {} / {}。".format(tuple(mask.shape), mask.dtype))
    if state is not None:
        _check_mamba3_state(state, layer, int(x.shape[0]), x.device)


def check_mamba3_step_inputs(x, state, layer):
    """校验对象: Mamba3.step 的 token 与 state 入参。"""
    if x.ndim != 2 or int(x.shape[-1]) != layer.d_model:
        raise ValueError("step 输入 x 期望 [B,d_model={}], 实际 {}。".format(layer.d_model, tuple(x.shape)))
    _check_mamba3_state(state, layer, int(x.shape[0]), x.device)
