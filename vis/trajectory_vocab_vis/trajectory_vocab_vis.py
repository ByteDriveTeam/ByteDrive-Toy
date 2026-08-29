"""绘制几何轨迹词表、速度词表和聚类评估图。

模块: vis/trajectory_vocab_vis/trajectory_vocab_vis.py
依赖: matplotlib, numpy
读取配置: trajectory_vocabulary.*
对外接口:
    - render_vocabulary(cfg) -> list[Path]
"""
from pathlib import Path
import numpy as np


def _load(path):
    return np.load(path, allow_pickle=True).item()


def render_vocabulary(cfg):
    """从 NPY 词表生成静态 PNG。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(cfg.output_dir); out = root / "visualization"; out.mkdir(parents=True, exist_ok=True)
    g, s = _load(root / cfg.geometry_output), _load(root / cfg.speed_output)
    paths = []
    fig, ax = plt.subplots(figsize=(10, 10)); curves = g["vocab"][:cfg.max_curves]
    for i, curve in enumerate(curves):
        ax.plot(np.r_[0, curve[:, 0]], np.r_[0, curve[:, 1]], alpha=0.45, linewidth=1)
        ax.text(curve[-1, 0], curve[-1, 1], str(i), fontsize=6)
    ax.scatter([0], [0], c="red", s=30); ax.set_aspect("equal"); ax.set_xlabel("前向 x (m)"); ax.set_ylabel("左/右 y (m)")
    ax.set_title("Ego 几何轨迹词表 (K={}, {})".format(len(g["vocab"]), g["algorithm"])); ax.grid(alpha=.2)
    p = out / "geometry_vocab.png"; fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig); paths.append(p)
    fig, ax = plt.subplots(figsize=(10, 6)); t = np.arange(s["vocab"].shape[1]) / cfg.speed_hz + 1.0 / cfg.speed_hz
    for i, curve in enumerate(s["vocab"][:cfg.max_curves]):
        ax.plot(t, curve, linewidth=2 if i == 0 else 1, alpha=.8, label="静止" if i == 0 else str(i))
    ax.set_xlabel("未来时间 (s)"); ax.set_ylabel("速度 (m/s)"); ax.set_title("Ego 速度词表 (K={})".format(len(s["vocab"]))); ax.grid(alpha=.2)
    ax.legend(fontsize=7, ncol=4); p = out / "speed_vocab.png"; fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig); paths.append(p)
    fig, ax = plt.subplots(figsize=(9, 5)); labels = ["geometry", "speed"]
    vals = [g["metrics"]["mean_nearest_distance"], s["metrics"]["mean_nearest_distance"]]
    ax.bar(labels, vals); ax.set_ylabel("平均最近词距离"); ax.set_title("词表覆盖度评估"); ax.grid(axis="y", alpha=.2)
    p = out / "metrics.png"; fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig); paths.append(p)
    return paths
