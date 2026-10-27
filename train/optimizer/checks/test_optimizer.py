"""优化器参数范围与免衰减分组的回归测试。

模块: train/optimizer/checks/test_optimizer.py
依赖: types, unittest, torch, train.optimizer
读取配置: —（数值仅为测试样例）
对外接口: —（通过 python -m unittest train.optimizer.checks.test_optimizer 运行）
"""

from types import SimpleNamespace
import unittest

import torch
from torch import nn

from train.optimizer import build_optimizer


def _config():
    return SimpleNamespace(train=SimpleNamespace(
        lr=0.01, weight_decay=0.1, fused_optimizer=False, perception_lr_scale=0.2))


class OptimizerTests(unittest.TestCase):
    """校验对象: build_optimizer —— 保留调用方分组并实际应用 _no_weight_decay 标记。"""

    def test_plain_module_weight_decay(self):
        """校验对象: 普通 nn.Module —— 过滤冻结参数，免衰减参数在零梯度下保持值。"""
        model = nn.Linear(3, 2)
        model.weight._no_weight_decay = True
        model.register_parameter("frozen", nn.Parameter(torch.ones(1), requires_grad=False))
        optimizer = build_optimizer(model, _config())
        groups = {id(p): group for group in optimizer.param_groups for p in group["params"]}
        self.assertNotIn(id(model.frozen), groups)
        self.assertEqual(groups[id(model.weight)]["weight_decay"], 0)
        self.assertEqual(groups[id(model.bias)]["weight_decay"], 0.1)
        weight, bias = model.weight.detach().clone(), model.bias.detach().clone()
        model.weight.grad, model.bias.grad = torch.zeros_like(weight), torch.zeros_like(bias)
        optimizer.step()
        torch.testing.assert_close(model.weight, weight, rtol=0, atol=0)
        torch.testing.assert_close(model.bias, bias * (1 - 0.01 * 0.1))

    def test_existing_groups_keep_options(self):
        """校验对象: 已有 param_groups —— 拆分后保留学习率、组内选项及默认衰减。"""
        class Grouped(nn.Module):
            def __init__(self):
                super().__init__()
                self.main = nn.Parameter(torch.ones(1))
                self.special = nn.Parameter(torch.ones(1))
                self.slow = nn.Parameter(torch.ones(1))
                self.special._no_weight_decay = True

            def param_groups(self, lr, decay, scale):
                return [{"params": [self.main, self.special], "lr": lr, "weight_decay": decay,
                         "betas": (0.8, 0.95)},
                        {"params": (p for p in [self.slow]), "lr": lr * scale}]

        model = Grouped()
        optimizer = build_optimizer(model, _config())
        groups = {id(p): group for group in optimizer.param_groups for p in group["params"]}
        self.assertEqual(len(groups), 3)
        self.assertEqual(groups[id(model.special)]["lr"], 0.01)
        self.assertEqual(groups[id(model.special)]["betas"], (0.8, 0.95))
        self.assertEqual(groups[id(model.special)]["weight_decay"], 0)
        self.assertEqual(groups[id(model.slow)]["lr"], 0.002)
        self.assertEqual(groups[id(model.slow)]["weight_decay"], 0.1)

    def test_trainable_parameter_selection(self):
        """校验对象: trainable_parameters —— 保持任务主动选择的参数集合。"""
        class Selected(nn.Linear):
            def trainable_parameters(self):
                return iter((self.weight,))

        model = Selected(3, 2)
        optimizer = build_optimizer(model, _config())
        selected = [p for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(selected), 1)
        self.assertIs(selected[0], model.weight)


if __name__ == "__main__":
    unittest.main()
