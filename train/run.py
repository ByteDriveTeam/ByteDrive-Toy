"""统一训练入口 CLI：按任务加载配置，构建模型与数据，逐 epoch 训练并保存检查点。

模块: train/run.py
依赖: argparse, pathlib, torch, config.load_config, model.perception_model.PerceptionModel,
      model.driving_model.DrivingModel, data.perception_dataset.PerceptionDataset,
      data.bevseg_cache, train.optimizer, train.loop, train.bevseg,
      train.privileged_bev_vae, train.bevseg_vae64, train.checks.run_checks
读取配置:
    train.device / train.compile / train.float32_matmul_precision /
        train.batch_size / train.num_workers / train.prefetch_factor / train.in_order /
        train.shuffle / train.drop_last / train.pin_memory / train.persistent_workers /
        train.ckpt_dir / train.resume / train.epochs
    data.bevseg.cache.prebuild / data.bevseg.cache.progress_every
    train.privileged_bev_vae.noise.enabled / train.bevseg_vae64.noise.enabled
    （优化器、数据集和 epoch 函数读取的其余配置由对应模块负责）
对外接口:
    - main(argv=None) -> None
说明: 支持 perception、driving、bevseg、privileged_bev_vae 和 bevseg_vae64 五项任务。
      相对检查点目录按任务分目录保存；检查点排除 `backbone.` 权重但保留优化器状态，支持兼容续训。
      driving uses the four-frame cache and never loads legacy perception checkpoints.
      设备和 DataLoader 行为由 config 控制，CUDA 不可用时回退 CPU；多 worker 入口受 `__main__` 守卫保护。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader, default_collate

from config import load_config
from data.bevseg_cache import prepare_bevseg_cache
from data.bevseg_dataset import BevSegDataset
from data.driving_cache import DrivingSliceDataset
from data.perception_dataset import PerceptionDataset
from data.scene_batch_sampler import SceneBatchSampler
from model.driving_model import DrivingModel
from model.perception_model import PerceptionModel
from model.bevseg_compressor import BEVSegCompressor
from model.privileged_bev_vae import PrivilegedBEVVAE
from model.bevseg_vae64 import BEVSegVAE64
from train.checks.run_checks import check_runtime
from train.loop import train_one_epoch
from train.unified_driving import train_unified_driving_epoch
from train.bevseg import train_bevseg_epoch
from train.privileged_bev_vae import train_privileged_bev_vae_epoch
from train.bevseg_vae64 import train_bevseg_vae64_epoch
from train.optimizer import build_optimizer
from tools.privileged_bev_vae_noise import measure_latent_noise

_CKPT_PATTERN = re.compile(r"epoch_(\d+)\.pt$")

_TASK_DESCRIPTIONS = {
    "perception": "单帧相机感知（语义与深度）",
    "driving": "four-frame monocular BEV/trajectory flow-matching driving",
    "bevseg": "BEVSeg 离散压缩器",
    "privileged_bev_vae": "特权 BEV 连续 latent VAE",
    "bevseg_vae64": "整幅 BEV 64x 连续 latent VAE",
}

# 任务名 → (模型类, 数据集类, epoch 函数)；公共装配、续训和保存逻辑只保留一份。
_TASKS = {
    "perception": (PerceptionModel, PerceptionDataset, train_one_epoch),
    "driving": (DrivingModel, DrivingSliceDataset, train_unified_driving_epoch),
    "bevseg": (BEVSegCompressor, BevSegDataset, train_bevseg_epoch),
    "privileged_bev_vae": (PrivilegedBEVVAE, BevSegDataset,
                            train_privileged_bev_vae_epoch),
    "bevseg_vae64": (BEVSegVAE64, BevSegDataset,
                      train_bevseg_vae64_epoch),
}
_BEV_TASKS = frozenset(("bevseg", "privileged_bev_vae", "bevseg_vae64"))


def _compact_collate(samples):
    """逐键拼批；常量视场掩码只保留一份，避免先堆叠再丢弃。"""
    first = samples[0]
    return {
        name: first[name] if name == "inview"
        else default_collate([sample[name] for sample in samples])
        for name in first
    }


def _compile_for_cuda(model, cfg, device) -> None:
    """仅在云端 CUDA 训练按配置原位编译，保持 state_dict/自定义方法接口不变。"""
    if not cfg.train.compile or device.type != "cuda":
        return
    if not hasattr(model, "compile"):
        print("[train] 当前 PyTorch 不支持 nn.Module.compile，继续 eager 训练")
        return
    model.compile()
    print("[train] 已启用 torch.compile（首次迭代包含编译冷启动）")


def _configure_cuda_math(cfg, device) -> None:
    """显式配置 CUDA FP32 matmul 精度，避免 Inductor 对未启用 TF32 的性能告警。"""
    if device.type != "cuda":
        return
    torch.set_float32_matmul_precision(cfg.train.float32_matmul_precision)
    print("[train] float32 matmul precision={}".format(
        cfg.train.float32_matmul_precision))


def _resolve_device(requested: str) -> torch.device:
    """按 config 请求选择设备；请求 cuda 但不可用则回退 cpu。"""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[train] CUDA 不可用，回退 CPU")
        return torch.device("cpu")
    return torch.device(requested)


def _resolve_ckpt_dir(ckpt_dir: str, task: str) -> Path:
    """把相对检查点目录解析到仓库根下，并按任务分子目录隔离。"""
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


def _maybe_resume(model, optimizer, ckpt_dir: Path, resume: bool, explicit) -> int:
    """按需从检查点恢复模型+优化器，返回起始 epoch（已完成的 epoch 数）。"""
    path = Path(explicit) if explicit else (_find_latest_checkpoint(ckpt_dir) if resume else None)
    if path is None:
        return 0
    ckpt = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
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



def _build_parser() -> argparse.ArgumentParser:
    """构造训练 CLI；参数默认值仍由配置文件提供。"""
    task_lines = "\n".join(
        "  {:<20} {}".format(name, _TASK_DESCRIPTIONS[name])
        for name in _TASKS)
    parser = argparse.ArgumentParser(
        prog="python -m train.run",
        description="ByteDrive 统一训练入口：构建数据、模型和优化器，按 epoch 保存检查点。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "任务说明:\n"
            + task_lines
            + "\n\n常用命令:\n"
              "  python -m train.run --task perception --env <name>\n"
              "  python -m train.run --task driving --env <name>\n"
              "  python -m train.run --task bevseg --prepare-bevseg-cache"
        ),
    )
    parser.add_argument(
        "--task", default="perception", choices=sorted(_TASKS),
        help="训练任务（默认：perception；完整任务说明见命令末尾）",
    )
    parser.add_argument(
        "--config", default=None, metavar="PATH",
        help="主配置文件路径（默认：config/default.yaml）",
    )
    parser.add_argument(
        "--env", default=None, metavar="NAME",
        help="环境覆盖名，叠加 config/<NAME>.yaml",
    )
    parser.add_argument(
        "--resume", default=None, metavar="PATH",
        help="显式恢复检查点；省略时按 train.resume 自动选择任务目录中的最新检查点",
    )
    parser.add_argument(
        "--perception-ckpt", default=None, metavar="PATH",
        help="Deprecated: the new DrivingModel does not load legacy perception checkpoints",
    )
    parser.add_argument(
        "--prepare-bevseg-cache", action="store_true",
        help="仅预热三个 BEV 任务共享的栅格缓存，然后退出",
    )
    return parser


def _validate_cli_args(parser: argparse.ArgumentParser, args) -> None:
    """校验参数之间的关系，避免无效选项被静默忽略。"""
    if args.prepare_bevseg_cache and args.task not in _BEV_TASKS:
        parser.error("--prepare-bevseg-cache 仅可与 bevseg、privileged_bev_vae 或 bevseg_vae64 一起使用")
    if args.perception_ckpt:
        parser.error("新 DrivingModel 不再加载旧 perception checkpoint")


def main(argv=None) -> None:
    """解析训练参数，构建对应任务并执行配置指定的 epoch 数。"""
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_cli_args(parser, args)

    cfg = load_config(args.config, args.env)
    model_cls, dataset_cls, epoch_fn = _TASKS[args.task]
    bev_task = args.task in _BEV_TASKS
    dataset = dataset_cls(cfg) if bev_task else None
    if bev_task and (args.prepare_bevseg_cache or cfg.data.bevseg.cache.prebuild):
        try:
            stats = prepare_bevseg_cache(
                dataset, cfg.train.num_workers, cfg.train.prefetch_factor,
                cfg.train.in_order, cfg.data.bevseg.cache.progress_every)
        except Exception:
            dataset.close()
            raise
        print("[bevseg-cache] namespace={} size={:.3f}/{:.3f} GiB full={} complete={} samples={}".format(
            stats["namespace"], stats["size_bytes"] / 1024 ** 3,
            stats["max_size_bytes"] / 1024 ** 3, stats["full"], stats["complete"],
            stats["samples"]), flush=True)
        if args.prepare_bevseg_cache:
            dataset.close()
            return

    device = _resolve_device(cfg.train.device)
    _configure_cuda_math(cfg, device)

    model = model_cls(cfg).to(device)
    if dataset is None:
        dataset = dataset_cls(cfg)
    check_runtime(model, dataset)
    # 驾驶训练：先以感知预训练权重初始化感知子模块（在续训覆盖之前）
    loader_kwargs = {
        "num_workers": cfg.train.num_workers,
        "pin_memory": cfg.train.pin_memory and device.type == "cuda",
        "persistent_workers": cfg.train.persistent_workers and cfg.train.num_workers > 0,
        "collate_fn": _compact_collate,
    }
    if args.task == "driving":
        loader_kwargs.update(batch_size=cfg.train.batch_size,
                             shuffle=cfg.train.shuffle, drop_last=cfg.train.drop_last)
    else:
        loader_kwargs["batch_sampler"] = SceneBatchSampler(
            dataset.frame_index, cfg.train.batch_size, cfg.train.shuffle, cfg.train.drop_last)
    if cfg.train.num_workers > 0:
        loader_kwargs.update(
            prefetch_factor=cfg.train.prefetch_factor,
            in_order=cfg.train.in_order)
    loader = DataLoader(dataset, **loader_kwargs)
    optimizer = build_optimizer(model, cfg)

    ckpt_dir = _resolve_ckpt_dir(cfg.train.ckpt_dir, args.task)
    start_epoch = _maybe_resume(model, optimizer, ckpt_dir, cfg.train.resume, args.resume)
    _compile_for_cuda(model, cfg, device)

    for epoch in range(start_epoch, cfg.train.epochs):
        stats = (epoch_fn(model, loader, optimizer, cfg, device, epoch=epoch)
                 if args.task in ("bevseg", "privileged_bev_vae", "bevseg_vae64")
                 else epoch_fn(model, loader, optimizer, cfg, device))
        print("[train:{}] epoch {}/{} {}".format(
            args.task, epoch + 1, cfg.train.epochs,
            "  ".join("{}={:.4f}".format(k, v) for k, v in stats.items())))
        _save_checkpoint(model, optimizer, ckpt_dir / "epoch_{:03d}.pt".format(epoch + 1), epoch + 1)
        noise_key = {"privileged_bev_vae": "privileged_bev_vae",
                     "bevseg_vae64": "bevseg_vae64"}.get(args.task)
        if noise_key and getattr(cfg.train, noise_key).noise.enabled:
            noise = measure_latent_noise(
                model, dataset, cfg, device, epoch=epoch, train_config_key=noise_key)
            print("[{}-noise] epoch={} effective_rank={:.3f} near_zero={}".format(
                args.task, noise["epoch"], noise["effective_rank"],
                noise["near_zero_count"]), flush=True)


if __name__ == "__main__":
    main()
