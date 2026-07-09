"""训练入口 CLI：加载配置 → 建模型/数据/优化器 → 逐 epoch 训练并保存权重。

模块: train/run.py
依赖: argparse, pathlib, torch, config.load_config, model.perception_model.PerceptionModel,
      data.perception_dataset.PerceptionDataset, train.optimizer, train.loop, train.run_checks
读取配置:
    train.device / epochs / batch_size / num_workers / ckpt_dir
    （其余训练/模型/数据参数由各构造件各自读取）
对外接口:
    - main(argv=None) -> None      # 命令行入口
说明: 全项目唯一的训练启动点。设备取 config，但 CUDA 不可用时回退 CPU（便于本地冒烟）。检查点只保存
      可训练部分（排除冻结骨干的 `backbone.` 权重），避免每次落盘几十 M 的 DINO 权重。num_workers>0 时
      DataLoader 在 worker 内惰性建 SceneReader（见 data/perception_dataset），故入口置于 __main__ 守卫下。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import load_config
from data.perception_dataset import PerceptionDataset
from model.perception_model import PerceptionModel
from train.loop import train_one_epoch
from train.optimizer import build_optimizer
from train.run_checks import check_runtime


def _resolve_device(requested: str) -> torch.device:
    """按 config 请求选择设备；请求 cuda 但不可用则回退 cpu。"""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[train] CUDA 不可用，回退 CPU")
        return torch.device("cpu")
    return torch.device(requested)


def _save_trainable(model: PerceptionModel, path: Path, epoch: int) -> None:
    """仅保存可训练权重（排除冻结骨干），检查点小而够用于恢复训练/推理头。"""
    trainable = {k: v for k, v in model.state_dict().items() if not k.startswith("backbone.")}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "model": trainable}, path)


def main(argv=None) -> None:
    """训练主流程。"""
    parser = argparse.ArgumentParser(description="ByteDrive 感知模型训练")
    parser.add_argument("--config", default=None, help="主配置文件路径（缺省用 config/default.yaml）")
    parser.add_argument("--env", default=None, help="环境覆盖名（叠加 config/<env>.yaml）")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.env)
    device = _resolve_device(cfg.train.device)

    model = PerceptionModel(cfg).to(device)
    dataset = PerceptionDataset(cfg)
    check_runtime(model, dataset)
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True,
                        num_workers=cfg.train.num_workers, drop_last=True, pin_memory=True)
    optimizer = build_optimizer(model, cfg)

    ckpt_dir = Path(cfg.train.ckpt_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = Path(__file__).resolve().parents[1] / ckpt_dir
    for epoch in range(cfg.train.epochs):
        stats = train_one_epoch(model, loader, optimizer, cfg, device)
        print("[train] epoch {}/{} {}".format(
            epoch + 1, cfg.train.epochs,
            "  ".join("{}={:.4f}".format(k, v) for k, v in stats.items())))
        _save_trainable(model, ckpt_dir / "epoch_{:03d}.pt".format(epoch + 1), epoch + 1)


if __name__ == "__main__":
    main()
