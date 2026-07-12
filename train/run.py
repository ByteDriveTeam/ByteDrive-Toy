"""训练入口 CLI：加载配置 → 建模型/数据/优化器 → （可选断点续训）逐 epoch 训练并保存权重。

模块: train/run.py
依赖: argparse, pathlib, torch, config.load_config, model.perception_model.PerceptionModel,
      data.perception_dataset.PerceptionDataset, train.optimizer, train.loop, train.run_checks
读取配置:
    train.device / epochs / batch_size / num_workers / ckpt_dir / resume
    （其余训练/模型/数据参数由各构造件各自读取）
对外接口:
    - main(argv=None) -> None      # 命令行入口
说明: 全项目唯一的训练启动点。设备取 config，但 CUDA 不可用时回退 CPU（便于本地冒烟）。检查点只保存
      可训练部分（排除冻结骨干的 `backbone.` 权重）加优化器状态与 epoch，故可断点续训又不落几十 M 的
      DINO 权重。resume（config，或 --resume 显式指定路径）为真时从最新/指定检查点恢复模型+优化器+起始
      epoch。num_workers>0 时 DataLoader 在 worker 内惰性建 SceneReader，故入口置于 __main__ 守卫下。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import load_config
from data.perception_dataset import PerceptionDataset
from model.perception_model import PerceptionModel
from train.loop import train_one_epoch
from train.optimizer import build_optimizer
from train.run_checks import check_runtime

_CKPT_PATTERN = re.compile(r"epoch_(\d+)\.pt$")


def _resolve_device(requested: str) -> torch.device:
    """按 config 请求选择设备；请求 cuda 但不可用则回退 cpu。"""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[train] CUDA 不可用，回退 CPU")
        return torch.device("cpu")
    return torch.device(requested)


def _resolve_ckpt_dir(ckpt_dir: str) -> Path:
    """把相对检查点目录解析到仓库根下。"""
    path = Path(ckpt_dir)
    return path if path.is_absolute() else Path(__file__).resolve().parents[1] / path


def _save_checkpoint(model: PerceptionModel, optimizer, path: Path, epoch: int) -> None:
    """保存可训练权重（排除冻结骨干）+ 优化器状态 + 已完成 epoch 数，供断点续训。"""
    trainable = {k: v for k, v in model.state_dict().items() if not k.startswith("backbone.")}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "model": trainable, "optimizer": optimizer.state_dict()}, path)


def _find_latest_checkpoint(ckpt_dir: Path) -> Path | None:
    """返回 ckpt_dir 下 epoch 序号最大的检查点；无则 None。"""
    if not ckpt_dir.is_dir():
        return None
    ckpts = [(int(m.group(1)), p) for p in ckpt_dir.glob("epoch_*.pt")
             if (m := _CKPT_PATTERN.search(p.name))]
    return max(ckpts)[1] if ckpts else None


def _maybe_resume(model: PerceptionModel, optimizer, ckpt_dir: Path,
                  resume: bool, explicit: str | None, device) -> int:
    """按需从检查点恢复模型+优化器，返回起始 epoch（已完成的 epoch 数）。"""
    path = Path(explicit) if explicit else (_find_latest_checkpoint(ckpt_dir) if resume else None)
    if path is None:
        return 0
    ckpt = torch.load(path, map_location=device)
    # 骨干权重不在检查点内，故 strict=False 容忍缺失的 backbone.* 键
    model.load_state_dict(ckpt["model"], strict=False)
    optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = int(ckpt["epoch"])
    print("[train] 从 {} 恢复，起始 epoch={}".format(path, start_epoch))
    return start_epoch


def main(argv=None) -> None:
    """训练主流程。"""
    parser = argparse.ArgumentParser(description="ByteDrive 感知模型训练")
    parser.add_argument("--config", default=None, help="主配置文件路径（缺省用 config/default.yaml）")
    parser.add_argument("--env", default=None, help="环境覆盖名（叠加 config/<env>.yaml）")
    parser.add_argument("--resume", default=None, help="显式指定要恢复的检查点路径（覆盖自动续训）")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.env)
    device = _resolve_device(cfg.train.device)

    model = PerceptionModel(cfg).to(device)
    dataset = PerceptionDataset(cfg)
    check_runtime(model, dataset)
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True,
                        num_workers=cfg.train.num_workers, drop_last=True, pin_memory=True)
    optimizer = build_optimizer(model, cfg)

    ckpt_dir = _resolve_ckpt_dir(cfg.train.ckpt_dir)
    start_epoch = _maybe_resume(model, optimizer, ckpt_dir, cfg.train.resume, args.resume, device)

    for epoch in range(start_epoch, cfg.train.epochs):
        stats = train_one_epoch(model, loader, optimizer, cfg, device)
        print("[train] epoch {}/{} {}".format(
            epoch + 1, cfg.train.epochs,
            "  ".join("{}={:.4f}".format(k, v) for k, v in stats.items())))
        _save_checkpoint(model, optimizer, ckpt_dir / "epoch_{:03d}.pt".format(epoch + 1), epoch + 1)


if __name__ == "__main__":
    main()
