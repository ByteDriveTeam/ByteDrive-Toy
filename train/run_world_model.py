"""世界模型单阶段联合训练 CLI：数据装配、AdamW、断点续训与检查点保存。

模块: train/run_world_model.py
依赖: argparse, pathlib, random, numpy, torch, config.load_config, data.world_model_dataset,
      model.world_model, train.world_model_loop
读取配置:
    train.device / fused_optimizer / float32_matmul_precision
    train.world_model.seed / batch_size / num_workers / prefetch_factor / in_order / shuffle /
        drop_last / pin_memory / persistent_workers / compile / amp_dtype / lr / weight_decay /
        adam_betas / adam_eps / ckpt_dir / resume / epochs
对外接口:
    - main(argv=None) -> None
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import load_config
from data.world_model_dataset import WorldModelDataset
from model.world_model import WorldModel
from train.world_model_loop import train_world_model_epoch


def main(argv=None) -> None:
    """执行配置定义的单阶段重建与 VISReg 联合训练。"""
    parser = argparse.ArgumentParser(description="训练 ByteDrive 掩码 BEV 世界模型")
    parser.add_argument("--config", default=None, help="主配置文件路径")
    parser.add_argument("--env", default=None, help="config/<env>.yaml 覆盖名")
    parser.add_argument("--resume", default=None, help="显式恢复检查点；缺省按配置使用 latest.pt")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.env)
    train_cfg = cfg.train.world_model
    _seed_all(train_cfg.seed)
    device = _device(cfg.train.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision(cfg.train.float32_matmul_precision)
    dataset = WorldModelDataset(cfg)
    loader = _loader(dataset, cfg, device)
    model = WorldModel(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay,
        betas=tuple(train_cfg.adam_betas), eps=train_cfg.adam_eps,
        fused=bool(cfg.train.fused_optimizer and device.type == "cuda"))
    ckpt_dir = _repo_path(train_cfg.ckpt_dir) / "world_model"
    state = _resume(model, optimizer, ckpt_dir, train_cfg.resume, args.resume)
    _maybe_compile(model, train_cfg, device)

    for epoch in range(state["epoch"], train_cfg.epochs):
        epoch_seed = train_cfg.seed + epoch
        stats, state["global_step"] = train_world_model_epoch(
            model, loader, optimizer, cfg, device, state["global_step"], epoch_seed)
        print("[world-model] epoch {}/{} {}".format(
            epoch + 1, train_cfg.epochs,
            "  ".join("{}={:.5g}".format(name, value) for name, value in stats.items())))
        _save(model, optimizer, ckpt_dir, epoch + 1, state["global_step"])
    dataset.close()


def _loader(dataset, cfg, device):
    train = cfg.train.world_model
    kwargs = {
        "batch_size": train.batch_size,
        "shuffle": train.shuffle,
        "drop_last": train.drop_last,
        "num_workers": train.num_workers,
        "pin_memory": train.pin_memory and device.type == "cuda",
        "persistent_workers": train.persistent_workers and train.num_workers > 0,
    }
    if train.num_workers > 0:
        kwargs.update(prefetch_factor=train.prefetch_factor, in_order=train.in_order)
    return DataLoader(dataset, **kwargs)


def _save(model, optimizer, directory, epoch, global_step):
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "epoch": epoch, "global_step": global_step,
    }
    temporary = directory / "latest.tmp.pt"
    torch.save(payload, temporary)
    os.replace(str(temporary), str(directory / "latest.pt"))


def _resume(model, optimizer, directory, enabled, explicit):
    path = Path(explicit) if explicit else (directory / "latest.pt" if enabled else None)
    state = {"epoch": 0, "global_step": 0}
    if path is None or not path.is_file():
        return state
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if "epoch" not in checkpoint:
        raise ValueError("检查点来自旧三阶段流程，不能推断其在新单阶段预算中的 epoch；"
                         "请关闭 resume 或显式提供新格式检查点")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    state.update({name: int(checkpoint[name]) for name in state})
    print("[world-model] 从 {} 恢复：epoch={} step={}".format(
        path, state["epoch"], state["global_step"]))
    return state


def _maybe_compile(model, train_cfg, device):
    if not train_cfg.compile or device.type != "cuda":
        return
    try:
        if os.name == "nt":
            # Windows 编译后端可能在首次真实 shape 才失败；开启抑制后由 Dynamo 自动回退 eager。
            torch._dynamo.config.suppress_errors = True
        model.student.compile()
        model.predictor.compile()
        print("[world-model] Student 与 Predictor 已启用 torch.compile")
    except Exception as error:
        print("[world-model] torch.compile 初始化失败，回退 eager：{}".format(error))


def _seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


def _device(requested):
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[world-model] CUDA 不可用，回退 CPU")
        return torch.device("cpu")
    return torch.device(requested)


def _repo_path(value):
    path = Path(value)
    return path if path.is_absolute() else Path(__file__).resolve().parents[1] / path


if __name__ == "__main__":
    main()
