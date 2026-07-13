"""训练入口 CLI：按 --task 选择感知/驾驶目标，加载配置 → 建模型/数据/优化器 → 逐 epoch 训练并保存权重。

模块: train/run.py
依赖: argparse, pathlib, torch, config.load_config, model.perception_model.PerceptionModel,
      model.driving_model.DrivingModel, data.perception_dataset.PerceptionDataset,
      data.driving_dataset.DrivingDataset, train.optimizer, train.loop, train.checks.run_checks
读取配置:
    train.device / epochs / batch_size / num_workers / ckpt_dir / resume
    （其余训练/模型/数据参数由各构造件各自读取）
对外接口:
    - main(argv=None) -> None      # 命令行入口
说明: 全项目唯一的训练启动点，感知与驾驶两条路径共用装配/续训/保存逻辑，仅模型/数据集/epoch 函数不同
      （--task 选择）。设备取 config，CUDA 不可用回退 CPU。检查点只保存非骨干权重（排除任何含 `backbone.`
      的键），故驾驶模型也不落几十 M 的 DINO 权重、可断点续训。驾驶训练可用 --perception-ckpt 以感知预训练权重
      初始化其感知子模块（复用深度/分割表征）。num_workers>0 时 DataLoader 在 worker 内惰性建 SceneReader，
      故入口置于 __main__ 守卫下。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import load_config
from data.driving_dataset import DrivingDataset
from data.perception_dataset import PerceptionDataset
from model.driving_model import DrivingModel
from model.perception_model import PerceptionModel
from train.checks.run_checks import check_runtime
from train.loop import train_driving_epoch, train_one_epoch
from train.optimizer import build_optimizer

_CKPT_PATTERN = re.compile(r"epoch_(\d+)\.pt$")

# 任务名 → (模型类, 数据集类, 训练 epoch 函数)；感知与驾驶共用其余装配逻辑
_TASKS = {
    "perception": (PerceptionModel, PerceptionDataset, train_one_epoch),
    "driving": (DrivingModel, DrivingDataset, train_driving_epoch),
}


def _resolve_device(requested: str) -> torch.device:
    """按 config 请求选择设备；请求 cuda 但不可用则回退 cpu。"""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[train] CUDA 不可用，回退 CPU")
        return torch.device("cpu")
    return torch.device(requested)


def _resolve_ckpt_dir(ckpt_dir: str, task: str) -> Path:
    """把相对检查点目录解析到仓库根下，并按 task 分子目录（感知/驾驶权重不混放）。"""
    path = Path(ckpt_dir)
    base = path if path.is_absolute() else Path(__file__).resolve().parents[1] / path
    return base / task


def _save_checkpoint(model, optimizer, path: Path, epoch: int) -> None:
    """保存非骨干权重（排除任何含 backbone. 的键）+ 优化器状态 + 已完成 epoch 数，供断点续训。"""
    trainable = {k: v for k, v in model.state_dict().items() if "backbone." not in k}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "model": trainable, "optimizer": optimizer.state_dict()}, path)


def _find_latest_checkpoint(ckpt_dir: Path):
    """返回 ckpt_dir 下 epoch 序号最大的检查点；无则 None。"""
    if not ckpt_dir.is_dir():
        return None
    ckpts = [(int(m.group(1)), p) for p in ckpt_dir.glob("epoch_*.pt")
             if (m := _CKPT_PATTERN.search(p.name))]
    return max(ckpts)[1] if ckpts else None


def _maybe_resume(model, optimizer, ckpt_dir: Path, resume: bool, explicit, device) -> int:
    """按需从检查点恢复模型+优化器，返回起始 epoch（已完成的 epoch 数）。"""
    path = Path(explicit) if explicit else (_find_latest_checkpoint(ckpt_dir) if resume else None)
    if path is None:
        return 0
    ckpt = torch.load(path, map_location=device)
    # 骨干权重不在检查点内，故 strict=False 容忍缺失的 *backbone.* 键
    model.load_state_dict(ckpt["model"], strict=False)
    optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = int(ckpt["epoch"])
    print("[train] 从 {} 恢复，起始 epoch={}".format(path, start_epoch))
    return start_epoch


def _load_perception_weights(model: DrivingModel, path, device) -> None:
    """以感知预训练权重初始化驾驶模型的感知子模块（融合+trunk+双头；骨干仍从 DINO 本地权重加载）。"""
    ckpt = torch.load(path, map_location=device)
    missing, unexpected = model.perception.load_state_dict(ckpt["model"], strict=False)
    print("[driving] 载入感知预训练权重 {}（缺失 {} 项，多余 {} 项）".format(
        path, len(missing), len(unexpected)))


def main(argv=None) -> None:
    """训练主流程（感知或驾驶）。"""
    parser = argparse.ArgumentParser(description="ByteDrive 训练（感知 / 驾驶）")
    parser.add_argument("--task", default="perception", choices=sorted(_TASKS),
                        help="训练目标：perception（默认）或 driving")
    parser.add_argument("--config", default=None, help="主配置文件路径（缺省用 config/default.yaml）")
    parser.add_argument("--env", default=None, help="环境覆盖名（叠加 config/<env>.yaml）")
    parser.add_argument("--resume", default=None, help="显式指定要恢复的检查点路径（覆盖自动续训）")
    parser.add_argument("--perception-ckpt", default=None,
                        help="驾驶训练时用于初始化感知子模块的感知检查点路径")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.env)
    device = _resolve_device(cfg.train.device)
    model_cls, dataset_cls, epoch_fn = _TASKS[args.task]

    model = model_cls(cfg).to(device)
    dataset = dataset_cls(cfg)
    check_runtime(model, dataset)
    # 驾驶训练：先以感知预训练权重初始化感知子模块（在续训覆盖之前）
    if args.task == "driving" and args.perception_ckpt:
        _load_perception_weights(model, args.perception_ckpt, device)
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True,
                        num_workers=cfg.train.num_workers, drop_last=True, pin_memory=True)
    optimizer = build_optimizer(model, cfg)

    ckpt_dir = _resolve_ckpt_dir(cfg.train.ckpt_dir, args.task)
    start_epoch = _maybe_resume(model, optimizer, ckpt_dir, cfg.train.resume, args.resume, device)

    for epoch in range(start_epoch, cfg.train.epochs):
        stats = epoch_fn(model, loader, optimizer, cfg, device)
        print("[train:{}] epoch {}/{} {}".format(
            args.task, epoch + 1, cfg.train.epochs,
            "  ".join("{}={:.4f}".format(k, v) for k, v in stats.items())))
        _save_checkpoint(model, optimizer, ckpt_dir / "epoch_{:03d}.pt".format(epoch + 1), epoch + 1)


if __name__ == "__main__":
    main()
