"""Mamba-3 官方计算定义与混合精度状态的回归测试。

模块: model/mamba3/checks/test_mamba3.py
依赖: dataclasses, math, unittest, torch, model.mamba3
读取配置: —（结构与数值仅为独立测试样例，不作为模型默认配置）
对外接口: —（通过 python -m unittest model.mamba3.checks.test_mamba3 运行）
说明: 使用独立 PyTorch 公式验证 head 归一化和旋转；不依赖官方 GPU kernel。
"""

from dataclasses import replace
import math
import unittest

import torch
import torch.nn.functional as F

from model.mamba3 import Mamba3, Mamba3State, mamba3_forward
from model.mamba3.mamba3 import heavy_tail_activation


def _layer(**overrides):
    options = dict(
        d_model=8, d_state=8, expand=1, headdim=2, ngroups=2,
        rope_fraction=0.5, dt_min=0.01, dt_max=0.1, dt_init_floor=1e-4,
        A_floor=1e-4, is_mimo=False, mimo_rank=1, is_outproj_norm=True,
        fuse_pregate_headwise_norm=False, chunk_size=3, norm_eps=1e-5,
    )
    options.update(overrides)
    return Mamba3(**options)


def _serial_reference(layer, x, mask, reset, initial):
    """独立串行公式；不复用模型的投影、group 扩展、旋转或状态更新辅助函数。"""
    phase, h, bp, xp = initial.angle, initial.ssm, initial.prev_b, initial.prev_x
    outputs = []
    heads, width, size, rank = layer.nheads, layer.headdim, layer.d_state, layer.mimo_rank
    group_index = torch.arange(heads, device=x.device) // (heads // layer.ngroups)
    sizes = (layer.d_inner, layer.d_inner, size * layer.ngroups * rank,
             size * layer.ngroups * rank, heads, heads, heads, layer.num_rope_angles)
    for t in range(x.shape[1]):
        def clear(value):
            return torch.where(reset[:, t].reshape(-1, *((1,) * (value.ndim - 1))), 0, value)

        phase, h, bp, xp = map(clear, (phase, h, bp, xp))
        token = torch.where(mask[:, t, None], x[:, t], 0)
        z, v, b, c, rd, ra, rt, rw = F.linear(token, layer.in_proj.weight).split(sizes, -1)
        z, v = z.reshape(-1, heads, width).float(), v.reshape(-1, heads, width).float()
        b = F.rms_norm(b.reshape(-1, rank, layer.ngroups, size).float(), (size,),
                       layer.B_norm_weight.float(), layer.norm_eps)
        c = F.rms_norm(c.reshape(-1, rank, layer.ngroups, size).float(), (size,),
                       layer.C_norm_weight.float(), layer.norm_eps)
        b = b[:, :, group_index] + layer.B_bias.float().permute(1, 0, 2)
        c = c[:, :, group_index] + layer.C_bias.float().permute(1, 0, 2)
        dt = F.softplus(rd.float() + layer.dt_bias.float())
        activation = torch.where(ra >= 0, 1 + ra.float(), 1 / (1 - ra.float().clamp_max(0)))
        a, trap = -activation.clamp_min(layer.A_floor), rt.float().sigmoid()
        next_phase = torch.remainder(phase + math.pi * rw.float().tanh()[:, None] * dt[..., None],
                                     2 * math.pi)

        def rotate(value):
            result = value.clone()
            for i in range(layer.num_rope_angles):
                j, k = (i, i + size // 2) if layer.is_mimo else (2 * i, 2 * i + 1)
                cosine, sine = next_phase[..., i].cos()[:, None], next_phase[..., i].sin()[:, None]
                result[..., j] = value[..., j] * cosine - value[..., k] * sine
                result[..., k] = value[..., j] * sine + value[..., k] * cosine
            return result

        b, c = rotate(b), rotate(c)

        def up(value, weight):
            return (value[:, None].float() * weight.float().permute(1, 0, 2)
                    if weight is not None else value[:, None].float())

        vr, vp = up(v, layer.mimo_x), up(xp, layer.mimo_x)
        alpha = (a * dt).exp()
        beta, gamma = (1 - trap) * dt * alpha, trap * dt
        source = (vr[..., None] * b[..., None, :]).sum(1)
        previous = (vp[..., None] * bp.float()[..., None, :]).sum(1)
        hn = h * alpha[..., None, None] + beta[..., None, None] * previous + gamma[..., None, None] * source
        yr = (hn[:, None] * c[..., None, :]).sum(-1) + layer.D.float()[None, None, :, None] * vr
        if layer.is_outproj_norm:
            yr = F.rms_norm(yr, (width,), eps=layer.norm_eps) * layer.out_norm_weight.float().view(heads, width)
        yr = yr * F.silu(up(z, layer.mimo_z))
        output = ((yr * layer.mimo_o.float().permute(1, 0, 2)).sum(1)
                  if layer.is_mimo else yr[:, 0])
        output = F.linear(output.flatten(1).to(layer.out_proj.weight.dtype), layer.out_proj.weight)
        outputs.append(torch.where(mask[:, t, None], output, 0))

        def update(old, new):
            return torch.where(mask[:, t].reshape(-1, *((1,) * (old.ndim - 1))), new.to(old.dtype), old)

        phase, h, bp, xp = update(phase, next_phase), update(h, hn), update(bp, b), update(xp, v)
    return torch.stack(outputs, 1), Mamba3State(phase, h, bp, xp)


class Mamba3Tests(unittest.TestCase):
    """校验对象: Mamba3 的输出定义、旋转布局及状态精度契约。"""

    def setUp(self):
        """固定随机样例以便复查数值误差。"""
        torch.manual_seed(1729)

    def test_heavy_tail_zero_derivative_and_gradcheck(self):
        """校验对象: heavy_tail_activation —— 零点导数为 1，未选择的分支也不产生奇点。"""
        x = torch.tensor([-2., -0.01, 0., 0.01, 1., 2.], dtype=torch.float64, requires_grad=True)
        y = heavy_tail_activation(x)
        grad, = torch.autograd.grad(y.sum(), x)
        expected = torch.where(x < 0, (1 - x.clamp_max(0)).reciprocal().square(), 1)
        torch.testing.assert_close(grad, expected, rtol=0, atol=1e-14)
        self.assertEqual(grad[2].item(), 1)
        self.assertTrue(torch.autograd.gradcheck(heavy_tail_activation, (x,), eps=1e-6, atol=1e-6))
        extremes = torch.tensor([-1e30, 0., 1., 1e30], requires_grad=True)
        heavy_tail_activation(extremes).sum().backward()
        self.assertTrue(extremes.grad.isfinite().all())

    def test_constructor_rejects_truncation_and_nonfinite_scalars(self):
        """校验对象: 构造参数 —— 拒绝浮点整数、静默截断、bool 数值和非有限/越界常量。"""
        integers = ("d_model", "d_state", "expand", "headdim", "ngroups", "mimo_rank", "chunk_size")
        for name in integers:
            for value in (0, -1, 1.5, 2.0, True, "2", float("nan"), float("inf")):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    _layer(**{name: value})
        for name in ("rope_fraction", "dt_min", "dt_max", "dt_init_floor", "A_floor", "norm_eps"):
            for value in (0, -1, float("nan"), float("inf"), float("-inf"), True, "0.1", 1e300, 1e-300):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    _layer(**{name: value})
        for name in ("is_mimo", "is_outproj_norm", "fuse_pregate_headwise_norm"):
            for value in (0, 1, "false", None):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    _layer(**{name: value})
        with self.assertRaises(ValueError):
            _layer(dt_min=1., dt_max=0.1)

    def test_stable_rmsnorm_large_finite_values_and_gradients(self):
        """校验对象: _rms_norm —— 大数、零值、小信号的输出和梯度对齐独立 FP64 公式。"""
        layer = _layer()
        value = torch.tensor([[1e20, 2e20], [3e38, -2e38], [0., 0.], [1e-20, -1e-20],
                              [1., -2.], [8., 8.]], requires_grad=True)
        weight = torch.tensor([0.8, 1.3], requires_grad=True)
        actual = layer._rms_norm(value, weight)
        expected = (value.double() * (value.double().square().mean(-1, keepdim=True)
                                      + layer.norm_eps).rsqrt() * weight.double())
        self.assertTrue(actual.isfinite().all())
        torch.testing.assert_close(actual.double(), expected, rtol=3e-6, atol=1e-7)
        got_grad = torch.autograd.grad(actual.sum(), (value, weight))
        ref_grad = torch.autograd.grad(expected.sum(), (value, weight))
        for got, want in zip(got_grad, ref_grad):
            self.assertTrue(got.isfinite().all())
            torch.testing.assert_close(got, want, rtol=5e-5, atol=1e-42)

    def test_mimo_compensated_d_skip_initialization(self):
        """校验对象: D 初值 —— 补偿 rank 后的初始直通增益与 SISO 对齐，加载权重不重补偿。"""
        x = torch.tensor([0.5, -1.2, 2., -0.7, 1., 0.1, -0.3, 1.7]).reshape(1, 1, 4, 2)
        z = torch.ones_like(x)
        states = torch.zeros(1, 1, 4, 2, 8)
        for normalized in (False, True):
            reference = None
            for rank in (1, 2, 3, 4, 8):
                with self.subTest(rank=rank, normalized=normalized):
                    layer = _layer(is_mimo=rank > 1, mimo_rank=rank, is_outproj_norm=normalized)
                    torch.testing.assert_close(layer.D, torch.full_like(layer.D, float(rank)))
                    with torch.no_grad():
                        layer.out_proj.weight.copy_(torch.eye(8))
                    output = layer._decode(states, torch.zeros(1, 1, rank, 4, 8), x, z)
                    if reference is None:
                        reference = output
                    else:
                        torch.testing.assert_close(output, reference, rtol=2e-6, atol=2e-7)
                    weights = layer.state_dict()
                    weights["D"] = torch.full_like(layer.D, 1.25)
                    layer.load_state_dict(weights)
                    torch.testing.assert_close(layer.D, torch.full_like(layer.D, 1.25))

    def test_checkpoint_entrypoints_preserve_module_hooks(self):
        """校验对象: 重计算入口 —— 标准调用、兼容入口及函数式入口均执行 pre/post hooks。"""
        layer = _layer(is_mimo=True, mimo_rank=3)
        x = torch.randn(2, 7, 8, requires_grad=True)
        expected = layer(x + 0.1) * 0.5
        ref_grad = torch.autograd.grad(expected.sum(), (x, *layer.parameters()))
        calls = []

        def pre_hook(module, args, kwargs):
            calls.append("pre")
            return (args[0] + 0.1, *args[1:]), kwargs

        def post_hook(module, args, output):
            calls.append("post")
            return output * 0.5

        pre = layer.register_forward_pre_hook(pre_hook, with_kwargs=True)
        post = layer.register_forward_hook(post_hook)
        try:
            for fn in (lambda: layer(x), lambda: layer(x, checkpoint_chunks=True),
                       lambda: layer.forward_checkpointed(x),
                       lambda: mamba3_forward(x, layer, checkpoint_chunks=True)):
                calls.clear()
                output = fn()
                grad = torch.autograd.grad(output.sum(), (x, *layer.parameters()))
                self.assertEqual(calls, ["pre", "post"])
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                for got, want in zip(grad, ref_grad):
                    torch.testing.assert_close(got, want, rtol=0, atol=0)
            with self.assertRaises(ValueError):
                layer(x, checkpoint_chunks=1)
        finally:
            pre.remove()
            post.remove()

    def test_headwise_norm_and_gradients(self):
        """校验对象: _decode —— 各 head 独立归一化后门控，两条路径梯度一致。"""
        for mimo, fused in ((False, False), (True, False), (True, True)):
            with self.subTest(mimo=mimo, fused=fused):
                layer = _layer(is_mimo=mimo, mimo_rank=3 if mimo else 1,
                               fuse_pregate_headwise_norm=fused)
                with torch.no_grad():
                    layer.out_norm_weight.copy_(torch.linspace(0.4, 1.8, 8))
                    if mimo:
                        layer.mimo_x.uniform_(0.1, 0.8)
                        layer.mimo_z.uniform_(0.2, 1.2)
                        layer.mimo_o.uniform_(0.2, 0.9)
                states = (torch.randn(2, 3, 4, 2, 8)
                          * torch.tensor([1., 4., 16., 64.])[None, None, :, None, None])
                states.requires_grad_()
                c = torch.randn(2, 3, layer.mimo_rank, 4, 8, requires_grad=True)
                x = torch.randn(2, 3, 4, 2, requires_grad=True)
                z = torch.randn(2, 3, 4, 2, requires_grad=True)
                y = (states.unsqueeze(2) * c.unsqueeze(-2)).sum(-1)
                xr, zr = x.unsqueeze(2), z.unsqueeze(2)
                if mimo:
                    xr = xr * layer.mimo_x.permute(1, 0, 2)[None, None]
                    zr = zr * layer.mimo_z.permute(1, 0, 2)[None, None]
                y = y + layer.D[None, None, None, :, None] * xr
                y = F.rms_norm(y, (layer.headdim,), eps=layer.norm_eps)
                y = y * layer.out_norm_weight.view(4, 2) * F.silu(zr)
                y = ((y * layer.mimo_o.permute(1, 0, 2)[None, None]).sum(2)
                     if mimo else y[:, :, 0])
                expected = F.linear(y.flatten(-2), layer.out_proj.weight)
                actual = layer._decode(states, c, x, z)
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
                inputs = (states, c, x, z, layer.out_norm_weight)
                actual_grad = torch.autograd.grad(actual.square().sum(), inputs, retain_graph=True)
                expected_grad = torch.autograd.grad(expected.square().sum(), inputs)
                for got, want in zip(actual_grad, expected_grad):
                    torch.testing.assert_close(got, want, rtol=1e-4, atol=2e-5)

    def test_rotary_layout_dtype_and_gradients(self):
        """校验对象: _rotate —— 完整/部分旋转的官方配对、未旋转维与输出 dtype。"""
        for mimo in (False, True):
            for fraction in (0.5, 1.0):
                for dtype in (torch.float32, torch.bfloat16, torch.float16):
                    with self.subTest(mimo=mimo, fraction=fraction, dtype=dtype):
                        layer = _layer(is_mimo=mimo, mimo_rank=3 if mimo else 1,
                                       rope_fraction=fraction, dtype=dtype)
                        value = torch.randn(2, 3, layer.mimo_rank, 4, 8,
                                            dtype=dtype, requires_grad=True)
                        phase = torch.randn(2, 3, 4, layer.num_rope_angles, requires_grad=True)
                        expected = value.float().clone()
                        for i in range(layer.num_rope_angles):
                            a, b = (i, i + 4) if mimo else (2 * i, 2 * i + 1)
                            cosine = phase[..., i].cos().unsqueeze(2)
                            sine = phase[..., i].sin().unsqueeze(2)
                            expected[..., a] = value[..., a].float() * cosine - value[..., b].float() * sine
                            expected[..., b] = value[..., a].float() * sine + value[..., b].float() * cosine
                        expected = expected.to(dtype)
                        actual = layer._rotate(value, phase)
                        self.assertEqual(actual.dtype, dtype)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        got_grad = torch.autograd.grad(actual.float().square().sum(), (value, phase),
                                                       retain_graph=True)
                        ref_grad = torch.autograd.grad(expected.float().square().sum(), (value, phase))
                        for got, want in zip(got_grad, ref_grad):
                            torch.testing.assert_close(got, want, rtol=0.02, atol=0.02)

    def test_phase_projection(self):
        """校验对象: _project/step —— 相位增量为 pi*tanh(raw_angle)*dt。"""
        layer = _layer()
        x = torch.randn(2, 4, 8)
        raw = layer.in_proj(x)[..., -layer.num_rope_angles:]
        expected = (math.pi * raw.float().tanh()).unsqueeze(-2).expand(-1, -1, 4, -1)
        projected = layer._project(x)
        torch.testing.assert_close(projected[-1], expected)
        state = layer.allocate_state(2)
        state.angle.fill_(0.7)
        _, next_state = layer.step(x[:, 0], state)
        torch.testing.assert_close(next_state.angle,
                                   torch.remainder(state.angle + expected[:, 0]
                                                   * projected[4][:, 0, :, None], 2 * math.pi))

    def test_low_precision_scan_step_and_training(self):
        """校验对象: forward/step —— 跨 chunk 连续状态、低精度缓存及 FP32 参数更新。"""
        for mimo in (False, True):
            for dtype in (torch.float32, torch.bfloat16, torch.float16):
                with self.subTest(mimo=mimo, dtype=dtype):
                    layer = _layer(is_mimo=mimo, mimo_rank=3 if mimo else 1, dtype=dtype)
                    x = torch.randn(2, 11, 8, dtype=dtype)
                    state = layer.allocate_state(2)
                    outputs = []
                    for token in x.unbind(1):
                        output, state = layer.step(token, state)
                        outputs.append(output)
                        self.assertEqual(state.prev_b.dtype, dtype)
                        self.assertEqual(state.prev_x.dtype, dtype)
                        self.assertEqual(state.angle.dtype, torch.float32)
                        self.assertEqual(state.ssm.dtype, torch.float32)
                    actual, final = layer(x, return_state=True)
                    split, carry = layer(x[:, :5], return_state=True)
                    tail, carry = layer(x[:, 5:], state=carry, return_state=True)
                    tol = 2e-5 if dtype == torch.float32 else (0.02 if dtype == torch.bfloat16 else 0.003)
                    torch.testing.assert_close(actual, torch.stack(outputs, 1), rtol=tol, atol=tol)
                    torch.testing.assert_close(actual, torch.cat((split, tail), 1), rtol=tol, atol=tol)
                    for name in Mamba3State.__dataclass_fields__:
                        for other in (state, carry):
                            self.assertEqual(getattr(final, name).dtype, getattr(other, name).dtype)
                            got, want = getattr(final, name), getattr(other, name)
                            if name == "angle":
                                # 周期边界两侧的相位可能数值接近 0 和 2π，比较实际旋转。
                                got = torch.stack((got.cos(), got.sin()))
                                want = torch.stack((want.cos(), want.sin()))
                            torch.testing.assert_close(got, want, rtol=tol, atol=tol)
                    actual.float().square().mean().backward()
                    for parameter in layer.parameters():
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                    if mimo:
                        before = layer.mimo_x.detach().clone()
                        torch.optim.SGD(layer.parameters(), lr=0.1).step()
                        self.assertFalse(torch.equal(before, layer.mimo_x))
                        for name in ("mimo_x", "mimo_z", "mimo_o"):
                            self.assertEqual(getattr(layer, name).dtype, torch.float32)
                            self.assertEqual(getattr(layer, name).grad.dtype, torch.float32)

    def test_mimo_initialization(self):
        """校验对象: MIMO 参数 —— 低精度构造仍保留 FP32 的 1/rank 初值。"""
        for dtype in (torch.bfloat16, torch.float16):
            layer = _layer(is_mimo=True, mimo_rank=3, dtype=dtype)
            self.assertEqual(layer.in_proj.weight.dtype, dtype)
            for name, fill in (("mimo_x", 1 / 3), ("mimo_z", 1), ("mimo_o", 1 / 3)):
                parameter = getattr(layer, name)
                self.assertEqual(parameter.dtype, torch.float32)
                torch.testing.assert_close(parameter, torch.full(parameter.shape, fill, dtype=torch.float32),
                                           rtol=0, atol=0)

    def test_phase_bounds_wrapping_and_gradients(self):
        """校验对象: _project/forward/step —— 极端投影有界，跨周期累积取模且保留梯度。"""
        for mimo in (False, True):
            with self.subTest(mimo=mimo):
                layer = _layer(is_mimo=mimo, mimo_rank=3 if mimo else 1,
                               rope_fraction=1.0, chunk_size=7)
                with torch.no_grad():
                    layer.in_proj.weight.zero_()
                    layer.dt_bias.zero_()
                    layer.in_proj.weight[-4:, 0] = torch.tensor([-1e30, -1., 1., 1e30])
                x = torch.zeros(1, 129, 8)
                x[..., 0] = 1
                projected = layer._project(x)
                rate, dt = projected[-1], projected[4]
                self.assertTrue(torch.isfinite(rate).all())
                self.assertLessEqual(rate.abs().max().item(), torch.tensor(math.pi).item())
                rate_grad, = torch.autograd.grad(rate[:, 0].sum(), layer.in_proj.weight,
                                                 retain_graph=True)
                self.assertTrue(torch.isfinite(rate_grad).all())
                self.assertGreater(rate_grad[-3:-1, 0].abs().min().item(), 0)
                initial = layer.allocate_state(1)
                initial.angle.fill_(0.37)
                mask = (torch.arange(129) % 5 != 0).unsqueeze(0)
                _, state = layer(x, mask=mask, state=initial, return_state=True)
                expected = torch.remainder(
                    initial.angle.double() + rate[:, 0].double() * dt[:, 0, :, None].double()
                    * mask.sum().item(), 2 * math.pi)
                torch.testing.assert_close(state.angle.double(), expected, rtol=0, atol=3e-5)
                self.assertTrue(((state.angle >= 0) & (state.angle < 2 * math.pi)).all())
                grad, = torch.autograd.grad(state.angle.sum(), layer.in_proj.weight)
                self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(grad[-3:-1, 0].abs().min().item(), 0)
                stepped = initial
                for token in x[:, mask[0]].unbind(1):
                    _, stepped = layer.step(token, stepped)
                    self.assertEqual(stepped.angle.dtype, torch.float32)
                    self.assertTrue(((stepped.angle >= 0) & (stepped.angle < 2 * math.pi)).all())
                torch.testing.assert_close(stepped.angle.double(), expected, rtol=0, atol=5e-5)
                torch.testing.assert_close(stepped.angle, state.angle, rtol=0, atol=5e-5)
                step_grad, = torch.autograd.grad(stepped.angle.sum(), layer.in_proj.weight)
                self.assertTrue(torch.isfinite(step_grad).all())
                self.assertGreater(step_grad[-3:-1, 0].abs().min().item(), 0)

    def test_cache_dtype_override_and_autocast(self):
        """校验对象: 状态缓存 —— 显式 dtype 和 autocast 不改变已分配缓存的精度。"""
        for cache_dtype in (torch.float32, torch.bfloat16, torch.float16):
            with self.subTest(cache_dtype=cache_dtype):
                layer = _layer(is_mimo=True, mimo_rank=3)
                x = torch.randn(2, 5, 8)
                state = layer.allocate_state(2, dtype=cache_dtype)
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    for token in x.unbind(1):
                        _, state = layer.step(token, state)
                        self.assertEqual(state.prev_b.dtype, cache_dtype)
                        self.assertEqual(state.prev_x.dtype, cache_dtype)
                    _, state = layer(x, state=state, return_state=True)
                    self.assertEqual(state.prev_b.dtype, cache_dtype)
                    self.assertEqual(state.prev_x.dtype, cache_dtype)

    def test_masked_scan_and_invalid_state(self):
        """校验对象: mask/状态校验 —— 空洞与全无效 chunk 不覆盖缓存，非法 dtype 在入口拦截。"""
        layer = _layer(is_mimo=True, mimo_rank=3, dtype=torch.bfloat16)
        x = torch.randn(1, 9, 8, dtype=torch.bfloat16)
        mask = torch.tensor([[True, False, True, False, False, False, True, False, True]])
        initial = layer.allocate_state(1)
        initial.prev_b.normal_()
        initial.prev_x.normal_()
        output, state = layer(x, mask=mask, state=initial, return_state=True)
        expected, compact = layer(x[:, mask[0]], state=initial, return_state=True)
        torch.testing.assert_close(output[:, mask[0]], expected, rtol=0.02, atol=0.02)
        self.assertEqual(output[:, ~mask[0]].count_nonzero().item(), 0)
        for name in Mamba3State.__dataclass_fields__:
            torch.testing.assert_close(getattr(state, name), getattr(compact, name), rtol=0.02, atol=0.02)
        _, unchanged = layer(x, mask=torch.zeros_like(mask), state=initial, return_state=True)
        for name in Mamba3State.__dataclass_fields__:
            torch.testing.assert_close(getattr(unchanged, name), getattr(initial, name), rtol=0, atol=0)
        for bad in (replace(initial, angle=initial.angle.half()),
                    replace(initial, prev_b=initial.prev_b.float())):
            with self.assertRaises(ValueError):
                layer.step(x[:, 0], bad)
            with self.assertRaises(ValueError):
                layer(x, state=bad)
        with self.assertRaises(ValueError):
            _layer(is_mimo=True, mimo_rank=3, d_state=5)

    def test_group_mapping_and_serial_reference(self):
        """校验对象: group 与分段扫描 —— 合法分组、reset、空洞 mask 的输出及梯度对齐独立公式。"""
        for mimo in (False, True):
            for heads, groups in ((4, 1), (4, 2), (4, 4), (6, 2), (6, 3), (8, 2)):
                with self.subTest(mimo=mimo, heads=heads, groups=groups):
                    layer = _layer(d_model=heads * 2, ngroups=groups, is_mimo=mimo,
                                   mimo_rank=3 if mimo else 1, is_outproj_norm=groups != 3)
                    x = torch.randn(2, 11, heads * 2, requires_grad=True)
                    mask = torch.tensor([[True, False, True, False, True, True, False, False, False, True, True],
                                         [True, True, False, True, False, False, False, False, False, False, False]])
                    reset = torch.zeros_like(mask)
                    reset[0, [2, 6, 9]], reset[1, [0, 7, 10]] = True, True
                    zero = layer.allocate_state(2)
                    initial = Mamba3State(*(torch.randn_like(getattr(zero, name)).requires_grad_()
                                            for name in zero.__dataclass_fields__))
                    actual, state = layer(x, mask=mask, reset=reset, state=initial, return_state=True)
                    expected, reference = _serial_reference(layer, x, mask, reset, initial)
                    torch.testing.assert_close(actual, expected, rtol=5e-5, atol=2e-6)
                    for name in state.__dataclass_fields__:
                        torch.testing.assert_close(getattr(state, name), getattr(reference, name),
                                                   rtol=5e-5, atol=2e-6)
                    inputs = (x, *layer.parameters(), initial.angle, initial.ssm, initial.prev_b, initial.prev_x)
                    got_grad = torch.autograd.grad(actual.square().mean() + state.ssm.square().mean(), inputs,
                                                   retain_graph=True)
                    ref_grad = torch.autograd.grad(expected.square().mean() + reference.ssm.square().mean(), inputs)
                    for got, want in zip(got_grad, ref_grad):
                        torch.testing.assert_close(got, want, rtol=2e-4, atol=3e-6)
                    stepped = initial
                    outputs = []
                    for t, token in enumerate(x.unbind(1)):
                        output, stepped = layer.step(token, stepped, mask=mask[:, t], reset=reset[:, t])
                        outputs.append(output)
                    torch.testing.assert_close(torch.stack(outputs, 1), expected, rtol=5e-5, atol=2e-6)

    def test_fp32_core_and_rank_contraction(self):
        """校验对象: 外积与读出 —— autocast 下仍用 FP32，rank 收缩保持公式与梯度。"""
        for mimo in (False, True):
            layer = _layer(is_mimo=mimo, mimo_rank=3 if mimo else 1)
            x = torch.full((1, 2, 4, 2), 40000., dtype=torch.float16, requires_grad=True)
            b = torch.full((1, 2, layer.mimo_rank, 4, 8), 2., dtype=torch.float16, requires_grad=True)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                actual = layer._outer_sum(x, b)
            xr = x.float().unsqueeze(2)
            if mimo:
                xr = xr * layer.mimo_x.float().permute(1, 0, 2)
            expected = (xr[..., None] * b.float().unsqueeze(-2)).sum(2)
            self.assertEqual(actual.dtype, torch.float32)
            self.assertTrue(actual.isfinite().all())
            torch.testing.assert_close(actual, expected)
            inputs = (x, b, layer.mimo_x) if mimo else (x, b)
            got_grad = torch.autograd.grad(actual.mean(), inputs, retain_graph=True)
            ref_grad = torch.autograd.grad(expected.mean(), inputs)
            for got, want in zip(got_grad, ref_grad):
                torch.testing.assert_close(got, want)
            states = torch.full((1, 2, 4, 2, 8), 1e5)
            c, v, z = torch.ones_like(b), torch.ones_like(x), torch.ones_like(x)
            for dtype in (torch.float16, torch.bfloat16):
                with torch.autocast("cpu", dtype=dtype):
                    output = layer._decode(states, c, v, z)
                self.assertTrue(output.isfinite().all())

    def test_parameter_conversion_preserves_fp32_values(self):
        """校验对象: _apply —— 核心参数及梯度移动设备时保持 FP32，避免先量化再恢复。"""
        layer = _layer(is_mimo=True, mimo_rank=3)
        with torch.no_grad():
            for parameter in layer.parameters(recurse=False):
                parameter.uniform_(0.999, 1.003)
                parameter.grad = torch.full_like(parameter, 0.001001)
        before = {name: (p.detach().clone(), p.grad.clone(), id(p))
                  for name, p in layer.named_parameters(recurse=False)}
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            layer.to(dtype=dtype)
            self.assertEqual(layer.in_proj.weight.dtype, dtype)
            self.assertEqual(layer.out_proj.weight.dtype, dtype)
            for name, parameter in layer.named_parameters(recurse=False):
                self.assertEqual(parameter.dtype, torch.float32)
                self.assertEqual(parameter.grad.dtype, torch.float32)
                self.assertEqual(id(parameter), before[name][2])
                torch.testing.assert_close(parameter, before[name][0], rtol=0, atol=0)
                torch.testing.assert_close(parameter.grad, before[name][1], rtol=0, atol=0)
        self.assertTrue(layer.dt_bias._no_weight_decay)
        self.assertTrue(layer.D._no_weight_decay)

    def test_compact_state_detach_reset_and_empty(self):
        """校验对象: 状态 API —— 独立紧凑存储、显式截断、部分 reset 与空序列。"""
        layer = _layer(chunk_size=8)
        for batch in (1, 2):
            with torch.no_grad():
                _, state = layer(torch.randn(batch, 16, 8), return_state=True)
            for name in state.__dataclass_fields__:
                value = getattr(state, name)
                self.assertEqual(value.untyped_storage().nbytes(), value.numel() * value.element_size())
        x = torch.randn(2, 5, 8, requires_grad=True)
        _, state = layer(x, return_state=True)
        clone, detached = state.clone(), state.detach()
        for name in state.__dataclass_fields__:
            self.assertTrue(getattr(clone, name).requires_grad)
            self.assertFalse(getattr(detached, name).requires_grad)
            self.assertNotEqual(getattr(state, name).data_ptr(), getattr(detached, name).data_ptr())
        reset = state.reset(torch.tensor([True, False]))
        for name in state.__dataclass_fields__:
            self.assertEqual(getattr(reset, name)[0].count_nonzero().item(), 0)
            torch.testing.assert_close(getattr(reset, name)[1], getattr(state, name)[1])
        empty, carried = layer(x[:, :0], state=state, return_state=True,
                               mask=torch.empty(2, 0, dtype=torch.bool), reset=torch.empty(2, 0, dtype=torch.bool))
        self.assertEqual(empty.shape, (2, 0, 8))
        for name in state.__dataclass_fields__:
            torch.testing.assert_close(getattr(carried, name), getattr(state, name))
        with self.assertRaises(ValueError):
            layer(x, reset=torch.zeros(2, dtype=torch.bool))
        with self.assertRaises(ValueError):
            layer.step(x[:, 0], state, reset=torch.zeros(2, dtype=torch.float32))

    def test_reset_clears_nonfinite_history_and_mask_preserves_phase(self):
        """校验对象: reset/mask —— 清除旧非有限状态，暂停时不改写外部提供的相位表示。"""
        layer = _layer()
        x = torch.randn(1, 7, 8)
        initial = layer.allocate_state(1)
        initial.angle.fill_(-0.7)
        _, paused = layer(x, mask=torch.zeros(1, 7, dtype=torch.bool), state=initial, return_state=True)
        torch.testing.assert_close(paused.angle, initial.angle, rtol=0, atol=0)
        invalid = Mamba3State(*(torch.full_like(getattr(initial, name), float("nan"))
                                for name in initial.__dataclass_fields__))
        reset = torch.zeros(1, 7, dtype=torch.bool)
        reset[:, 0] = True
        actual, state = layer(x, state=invalid, reset=reset, return_state=True)
        expected, clean = layer(x, return_state=True)
        torch.testing.assert_close(actual, expected)
        for name in state.__dataclass_fields__:
            torch.testing.assert_close(getattr(state, name), getattr(clean, name))

    def test_nonfinite_padding_and_diagnostics(self):
        """校验对象: 无效输入与诊断 —— NaN/Inf padding 不污染有效输出、状态或梯度。"""
        layer = _layer(is_mimo=True, mimo_rank=3)
        x = torch.randn(2, 7, 8, requires_grad=True)
        mask = torch.tensor([[True, False, True, False, True, True, False],
                             [False, True, False, True, False, True, True]])
        with torch.no_grad():
            x[~mask] = float("nan")
            x[0, 1] = float("inf")
        output, state = layer(x, mask=mask, return_state=True)
        self.assertTrue(output.isfinite().all())
        self.assertTrue(state.ssm.isfinite().all())
        self.assertEqual(output[~mask].count_nonzero().item(), 0)
        output.square().mean().backward()
        self.assertTrue(x.grad.isfinite().all())
        self.assertEqual(x.grad[~mask].count_nonzero().item(), 0)
        for parameter in layer.parameters():
            self.assertTrue(parameter.grad.isfinite().all())
        before = state.detach()
        metrics = layer.diagnostics(x, mask=mask, state=state)
        self.assertEqual(metrics["valid_tokens"], mask.sum().item())
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
        self.assertLessEqual(metrics["dt_min"], metrics["dt_p95"])
        self.assertLessEqual(metrics["dt_p95"], metrics["dt_max"])
        self.assertEqual(layer.diagnostics(x, mask=torch.zeros_like(mask)), {"valid_tokens": 0})
        for name in state.__dataclass_fields__:
            torch.testing.assert_close(getattr(state, name), getattr(before, name))

    def test_checkpointed_chunks_preserve_gradients_and_reduce_saved_tensors(self):
        """校验对象: forward_checkpointed —— 保留跨调用梯度，降低 autograd 保存量。"""
        layer = _layer(is_mimo=True, mimo_rank=3, chunk_size=8)
        x = torch.randn(2, 32, 8, requires_grad=True)
        initial = layer.allocate_state(2)
        initial.ssm.requires_grad_()
        mask = torch.rand(2, 32) > 0.2
        reset = torch.zeros_like(mask)
        reset[:, 15] = True
        records = []
        for fn in (layer.forward, layer.forward_checkpointed):
            saved = []

            def pack(tensor):
                saved.append(tensor.numel() * tensor.element_size())
                return tensor

            with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                output, state = fn(x, mask=mask, reset=reset, state=initial, return_state=True)
                loss = output.square().mean() + state.ssm.square().mean()
            grad = torch.autograd.grad(loss, (x, initial.ssm, *layer.parameters()))
            records.append((output, state, grad, sum(saved)))
        torch.testing.assert_close(records[0][0], records[1][0], rtol=0, atol=0)
        for name in initial.__dataclass_fields__:
            torch.testing.assert_close(getattr(records[0][1], name), getattr(records[1][1], name), rtol=0, atol=0)
        for got, want in zip(records[0][2], records[1][2]):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        self.assertLess(records[1][3], records[0][3])

    def test_checkpointed_amp_and_optimizer_integration(self):
        """校验对象: AMP、重计算和优化器接入 —— 支持无输入梯度的训练与免衰减参数。"""
        from types import SimpleNamespace
        from train.optimizer import build_optimizer

        for dtype in (torch.bfloat16, torch.float16):
            layer = _layer(is_mimo=True, mimo_rank=3)
            x = torch.randn(2, 7, 8)
            mask = torch.rand(2, 7) > 0.2
            reset = torch.zeros_like(mask)
            reset[:, 3] = True
            with torch.autocast("cpu", dtype=dtype):
                expected, reference = layer(x, mask=mask, reset=reset, return_state=True)
                actual, state = layer.forward_checkpointed(x, mask=mask, reset=reset, return_state=True)
                loss = actual.float().square().mean()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for name in state.__dataclass_fields__:
                torch.testing.assert_close(getattr(state, name), getattr(reference, name), rtol=0, atol=0)
            loss.backward()
            self.assertTrue(all(p.grad.isfinite().all() for p in layer.parameters()))
            cfg = SimpleNamespace(train=SimpleNamespace(
                lr=0.01, weight_decay=0.1, fused_optimizer=False, perception_lr_scale=1.0))
            optimizer = build_optimizer(layer, cfg)
            decay = {id(p): group["weight_decay"] for group in optimizer.param_groups for p in group["params"]}
            self.assertEqual(decay[id(layer.dt_bias)], 0)
            self.assertEqual(decay[id(layer.D)], 0)
            self.assertEqual(decay[id(layer.in_proj.weight)], 0.1)
            optimizer.step()


if __name__ == "__main__":
    unittest.main()
