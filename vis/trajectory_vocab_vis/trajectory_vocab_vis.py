"""按词表内容绘制分解词表或耦合动作词表。

模块: vis/trajectory_vocab_vis/trajectory_vocab_vis.py
依赖: matplotlib, numpy, vis.trajectory_vocab_vis.checks
读取配置: trajectory_vocabulary.output_dir, action_output, geometry_output,
    speed_output, speed_hz, max_curves
对外接口:
    - render_vocabulary(cfg, vocab_path=None) -> list[Path]
"""
from pathlib import Path

import numpy as np

from vis.trajectory_vocab_vis.checks.trajectory_vocab_vis_checks import (
    check_action_bundle, check_bundle,
)


def _load_split(path):
    bundle = np.load(path, allow_pickle=True).item()
    check_bundle(bundle)
    return bundle


def _load_action(path):
    with np.load(path, allow_pickle=False) as saved:
        bundle = {name: saved[name] for name in saved.files}
    check_action_bundle(bundle)
    return bundle


def _resolve_vocabulary(cfg, vocab_path):
    """按文件内容识别词表；目录同时含两类词表时优先使用耦合动作词表。"""
    source = Path(vocab_path if vocab_path is not None else cfg.output_dir)
    if not source.exists():
        raise FileNotFoundError("词表路径不存在: {}".format(source))
    root = source if source.is_dir() else source.parent
    if source.is_file() and source.suffix == ".npz":
        return "action", _load_action(source), root
    if source.is_dir():
        candidates = [root / cfg.action_output] + sorted(root.glob("*.npz"))
        for path in dict.fromkeys(candidates):
            if path.is_file():
                try:
                    return "action", _load_action(path), root
                except (KeyError, ValueError):
                    continue
    elif source.suffix != ".npy":
        raise ValueError("只支持耦合词表 NPZ 或分解词表 NPY: {}".format(source))
    candidates = ([source] if source.is_file() else []) + [
        root / cfg.geometry_output, root / cfg.speed_output,
    ] + sorted(root.glob("*.npy"))
    bundles = {}
    for path in dict.fromkeys(candidates):
        if path.is_file():
            try:
                bundle = _load_split(path)
            except (ValueError, TypeError):
                if path == source:
                    raise
                continue
            bundles.setdefault(bundle["kind"], bundle)
    if set(bundles) != {"geometry", "speed"}:
        raise ValueError("分解词表需要同目录下的 geometry 和 speed 两个 NPY 文件: {}".format(root))
    return "split", bundles, root


def _selected_indices(count, limit):
    return np.linspace(0, count - 1, min(count, limit), dtype=np.int64)


def _save(fig, out, name, plt):
    path = out / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _render_action(bundle, out, max_curves, plt):
    """使用词的源轨迹展示几何形状，并直接绘制两个加速度通道。"""
    actions = bundle["actions"]
    hz = float(bundle["hz"])
    indices = _selected_indices(len(actions), max_curves)
    selected = actions[indices]
    times = np.arange(1, actions.shape[1] + 1) / hz
    paths = []
    if "source_trajectory" in bundle:
        fig, ax = plt.subplots(figsize=(10, 10))
        for curve in bundle["source_trajectory"][indices]:
            ax.plot(np.r_[0, curve[:, 0]], np.r_[0, curve[:, 1]], alpha=0.4, linewidth=0.8)
        ax.scatter([0], [0], c="red", s=30)
        ax.set_aspect("equal")
        ax.set(xlabel="局部前向 x (m)", ylabel="局部右向 y (m)",
               title="耦合动作词的源轨迹 (K={}，各自原始初速)".format(len(actions)))
        ax.grid(alpha=0.2)
        paths.append(_save(fig, out, "action_trajectories.png", plt))

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for component, ax in enumerate(axes):
        for curve in selected:
            ax.plot(times, curve[:, component], alpha=0.3, linewidth=0.8)
        ax.set_ylabel(("前向", "侧向")[component] + "加速度 (m/s²)")
        ax.grid(alpha=0.2)
    axes[0].set_title("耦合动作词表 (K={}，显示 {} 词)".format(len(actions), len(selected)))
    axes[-1].set_xlabel("未来时间 (s)")
    paths.append(_save(fig, out, "action_accelerations.png", plt))
    return paths


def _render_split(bundles, out, cfg, plt):
    geometry, speed = bundles["geometry"], bundles["speed"]
    curves = geometry["vocab"]
    fig, ax = plt.subplots(figsize=(10, 10))
    for index in _selected_indices(len(curves), cfg.max_curves):
        curve = curves[index]
        ax.plot(np.r_[0, curve[:, 0]], np.r_[0, curve[:, 1]], alpha=0.45, linewidth=1)
        if len(curves) <= 32:
            ax.text(curve[-1, 0], curve[-1, 1], str(index), fontsize=6)
    ax.scatter([0], [0], c="red", s=30)
    ax.set_aspect("equal")
    ax.set(xlabel="前向 x (m)", ylabel="左/右 y (m)",
           title="Ego 几何轨迹词表 (K={}, {})".format(len(curves), geometry["algorithm"]))
    ax.grid(alpha=0.2)
    paths = [_save(fig, out, "geometry_vocab.png", plt)]

    curves = speed["vocab"]
    times = np.arange(1, curves.shape[1] + 1) / cfg.speed_hz
    fig, ax = plt.subplots(figsize=(10, 6))
    for index in _selected_indices(len(curves), cfg.max_curves):
        ax.plot(times, curves[index], linewidth=1, alpha=0.8)
    ax.set(xlabel="未来时间 (s)", ylabel="速度 (m/s)",
           title="Ego 速度词表 (K={})".format(len(curves)))
    ax.grid(alpha=0.2)
    paths.append(_save(fig, out, "speed_vocab.png", plt))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(["geometry", "speed"], [
        geometry["metrics"]["mean_nearest_distance"],
        speed["metrics"]["mean_nearest_distance"],
    ])
    ax.set(ylabel="平均最近词距离", title="词表覆盖度评估")
    ax.grid(axis="y", alpha=0.2)
    paths.append(_save(fig, out, "metrics.png", plt))
    return paths


def render_vocabulary(cfg, vocab_path=None):
    """根据指定文件或目录中的词表内容生成 PNG。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    kind, bundle, root = _resolve_vocabulary(cfg, vocab_path)
    out = root / "visualization"
    out.mkdir(parents=True, exist_ok=True)
    if kind == "action":
        return _render_action(bundle, out, cfg.max_curves, plt)
    return _render_split(bundle, out, cfg, plt)
