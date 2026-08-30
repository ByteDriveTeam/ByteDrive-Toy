"""监控全局/逐参数梯度范数，拦截 NaN/Inf 并报告爆炸或消失风险。

模块: train/gradient_monitor/gradient_monitor.py
依赖: torch, config.schema.Config
读取配置:
    train.world_model.grad_monitor.enabled / log_every / explosion_norm / vanishing_norm / top_k
对外接口:
    - monitor_gradients(model, cfg, step) -> dict
"""

import torch

from config.schema import Config


__all__ = ["monitor_gradients"]


def monitor_gradients(model, cfg: Config, step: int) -> dict:
    """检查本优化步梯度；非有限值直接失败，其余按配置打印风险和最大参数。"""
    monitor = cfg.train.world_model.grad_monitor
    if not monitor.enabled:
        return {}
    named = [(name, parameter.grad) for name, parameter in model.named_parameters()
             if parameter.requires_grad and parameter.grad is not None]
    if not named:
        raise RuntimeError("梯度监控未发现任何可训练参数梯度")
    norms = torch.stack([gradient.detach().float().norm() for _, gradient in named])
    if not bool(torch.isfinite(norms).all()):
        bad = [name for (name, _), finite in zip(named, torch.isfinite(norms).tolist()) if not finite]
        raise FloatingPointError("检测到 NaN/Inf 梯度：{}".format(bad[:monitor.top_k]))
    global_norm = norms.square().sum().sqrt()
    value = float(global_norm)
    should_log = monitor.log_every > 0 and step % monitor.log_every == 0
    risky = value > monitor.explosion_norm or value < monitor.vanishing_norm
    if should_log or risky:
        count = min(int(monitor.top_k), len(named))
        top = torch.topk(norms, count).indices.tolist() if count else []
        details = ", ".join("{}={:.3e}".format(named[index][0], float(norms[index])) for index in top)
        status = "EXPLODING" if value > monitor.explosion_norm else (
            "VANISHING" if value < monitor.vanishing_norm else "OK")
        print("[grad] step={} global={:.3e} status={} top=[{}]".format(step, value, status, details))
    return {"gradient_norm": global_norm.detach()}
