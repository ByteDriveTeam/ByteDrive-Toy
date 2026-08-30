"""梯度健康监控模块的稳定公开入口。

模块: train/gradient_monitor/__init__.py
依赖: train.gradient_monitor.gradient_monitor
读取配置: —（实现通过 cfg 读取）
对外接口:
    - monitor_gradients(model, cfg, step) -> dict
"""

from train.gradient_monitor.gradient_monitor import monitor_gradients

__all__ = ["monitor_gradients"]
