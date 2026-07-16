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
    torch.save({"epoch": epoch, "model": trainable, "optimizer": optimizer.state_dict(),
                "optimizer_param_names": _optimizer_param_names(model, optimizer)}, path)


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
    loaded_names = _load_compatible_model(model, ckpt["model"])
    _load_compatible_optimizer(
        model, optimizer, ckpt.get("optimizer"), ckpt.get("optimizer_param_names"), loaded_names)
    start_epoch = int(ckpt["epoch"])
    print("[train] 从 {} 恢复，起始 epoch={}".format(path, start_epoch))
    return start_epoch


def _optimizer_param_names(model, optimizer):
    """按优化器参数组保存稳定参数名，供结构扩展后精确迁移动量。"""
    name_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    return [[name_by_id[id(parameter)] for parameter in group["params"]]
            for group in optimizer.param_groups]


def _load_compatible_model(model, saved_state):
    """载入键名与形状均兼容的模型权重；新增或改形状参数保留构造期初始化。"""
    current = model.state_dict()
    compatible = {name: value for name, value in saved_state.items()
                  if name in current and tuple(value.shape) == tuple(current[name].shape)}
    skipped = [name for name, value in saved_state.items()
               if name not in current or tuple(value.shape) != tuple(current[name].shape)]
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    initialized = [name for name in missing if "backbone." not in name]
    print("[train] 模型兼容恢复：载入 {} 项，新增初始化 {} 项，跳过 {} 项".format(
        len(compatible), len(initialized), len(skipped) + len(unexpected)))
    if initialized:
        print("[train]   新增/未覆盖参数保留自动初始化：{}".format(initialized[:8]))
    if skipped or unexpected:
        print("[train]   ⚠ 不兼容或多余参数：{}".format((skipped + list(unexpected))[:8]))
    return set(compatible)


def _load_compatible_optimizer(model, optimizer, saved, saved_name_groups, loaded_names):
    """把旧优化器状态迁移到已恢复参数；新增参数保持无动量状态并在首次 step 时自动建立。"""
    if not saved:
        print("[train] 检查点无优化器状态，使用新优化器")
        return
    current = optimizer.state_dict()
    if len(saved["param_groups"]) != len(current["param_groups"]):
        print("[train] ⚠ 优化器参数组数量变化，模型已恢复但优化器使用新状态")
        return

    current_names = _optimizer_param_names(model, optimizer)
    old_ids_by_name = _old_optimizer_ids_by_name(
        saved["param_groups"], saved_name_groups, current_names, loaded_names)
    if old_ids_by_name is None:
        print("[train] ⚠ 旧优化器参数无法可靠对齐，模型已恢复但优化器使用新状态")
        return

    new_state = {}
    restored_names = []
    new_names = []
    lazy_names = []
    for names, current_group in zip(current_names, current["param_groups"]):
        for name, current_id in zip(names, current_group["params"]):
            old_id = old_ids_by_name.get(name)
            if old_id in saved["state"]:
                new_state[current_id] = saved["state"][old_id]
                restored_names.append(name)
            elif old_id is None:
                new_names.append(name)
            else:
                lazy_names.append(name)
    groups = []
    for old_group, current_group in zip(saved["param_groups"], current["param_groups"]):
        merged = dict(current_group)
        merged.update({key: value for key, value in old_group.items() if key != "params"})
        merged["params"] = current_group["params"]
        groups.append(merged)
    optimizer.load_state_dict({"state": new_state, "param_groups": groups})
    total = sum(len(group["params"]) for group in current["param_groups"])
    print("[train] 优化器兼容恢复：迁移 {}/{} 个参数状态；新增无旧状态 {} 项；旧档未建状态 {} 项".format(
        len(restored_names), total, len(new_names), len(lazy_names)))
    if new_names:
        print("[train]   新增参数首次 step 自动建立状态：{}".format(new_names[:8]))
    if lazy_names:
        print("[train]   checkpoint 中未建立 AdamW 状态：{}".format(lazy_names[:8]))


def _old_optimizer_ids_by_name(old_groups, saved_name_groups, current_names, loaded_names):
    """优先按保存名映射；旧格式兼容新增参数及驾驶感知头从组尾移除前的稳定顺序。"""
    if saved_name_groups is not None:
        if len(saved_name_groups) != len(old_groups) or any(
                len(names) != len(group["params"])
                for names, group in zip(saved_name_groups, old_groups)):
            return None
        pairs = [(name, param_id)
                 for names, group in zip(saved_name_groups, old_groups)
                 for name, param_id in zip(names, group["params"])
                 if name in loaded_names]
        return dict(pairs)

    mapping = {}
    for old_group, names in zip(old_groups, current_names):
        reusable = [name for name in names if name in loaded_names]
        old_ids = old_group["params"]
        if len(reusable) > len(old_ids):
            return None
        if len(reusable) < len(old_ids) and not _is_legacy_feature_group(reusable):
            return None
        # 无参数名的旧驾驶优化器按 fusion、trunk、semantic_head、depth_head 排序；
        # 当前只保留前两者，故多出的旧 ID 必然是组尾两个未参与驾驶前向的感知头。
        mapping.update(zip(reusable, old_ids[:len(reusable)]))
    return mapping


def _is_legacy_feature_group(names):
    """判断当前组是否是旧驾驶优化器中位于两个感知解码头之前的 fusion/trunk 前缀。"""
    prefixes = ("perception.fusion.", "perception.trunk.")
    return bool(names) and all(name.startswith(prefixes) for name in names)


def _load_perception_weights(model: DrivingModel, path, device) -> None:
    """以感知预训练权重初始化驾驶模型的感知子模块（融合+trunk+双头；骨干仍从 DINO 本地权重加载）。

    感知检查点不含 DINOv3 骨干（保存时排除 backbone.* 键），故 strict=False 下「缺失」的必然是骨干参数——
    属正常，由本地 DINO 权重单独加载；只有「非骨干缺失」或「多余键」才是检查点与模型不匹配的真问题。
    """
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model", ckpt)  # 兼容 {epoch,model,optimizer} 或纯 state_dict
    missing, unexpected = model.perception.load_state_dict(state, strict=False)
    non_backbone_missing = [k for k in missing if "backbone." not in k]

    print("[driving] 载入感知预训练权重 {}：融合/trunk/双头 {} 项已载入".format(path, len(state)))
    if not non_backbone_missing:
        print("[driving]   缺失的 {} 项均为 DINOv3 骨干（由本地权重加载），属正常。".format(len(missing)))
    else:
        print("[driving]   ⚠ {} 个非骨干参数未被覆盖（检查点可能不完整）：{}".format(
            len(non_backbone_missing), non_backbone_missing[:5]))
    if unexpected:
        print("[driving]   ⚠ 检查点含 {} 个模型中无对应的多余键：{}".format(
            len(unexpected), unexpected[:5]))


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
