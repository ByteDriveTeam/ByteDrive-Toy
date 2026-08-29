"""从 CARLA ego 运动学时间轴构造并聚类几何/速度词表。

模块: data/trajectory_vocabulary/trajectory_vocabulary.py
依赖: numpy, vis.data_vis.reader
读取配置: trajectory_vocabulary.*
对外接口:
    - generate_vocabulary(cfg) -> dict
说明: 先按 0.5 秒重采样，再以块状计算特征和距离，避免保存全量高频时间轴及距离矩阵。
"""

from pathlib import Path
import time

import numpy as np

from vis.data_vis.reader import SceneReader, list_scenes
from vis.data_vis.geometry import transform_points, world_to_ego


def _size(value, cfg, n):
    if value != "auto":
        return min(int(value), n)
    candidates = range(cfg.auto_min_size, cfg.auto_max_size + 1, cfg.auto_step)
    return min(max(candidates), n)


def _resample(kinematics, step):
    t = np.asarray([x["sim_time"] for x in kinematics], dtype=np.float64)
    p = np.asarray([x["ego"]["transform"][:3] for x in kinematics], dtype=np.float64)
    v = np.asarray([np.linalg.norm(np.asarray(x["ego"]["velocity"][:2], dtype=np.float64))
                    for x in kinematics], dtype=np.float64)
    if len(t) < 2 or t[-1] <= t[0]:
        return np.empty((0,)), np.empty((0, 3)), np.empty((0,))
    grid = np.arange(t[0], t[-1] + step * 0.25, step, dtype=np.float64)
    grid = grid[grid <= t[-1] + 1e-9]
    return grid, np.column_stack([np.interp(grid, t, p[:, j]) for j in range(3)]), np.interp(grid, t, v)


def _window_features(kin, cfg):
    t, pos, speed = _resample(kin, cfg.motion_resample_s)
    if len(t) < 3:
        return np.empty((0, 50, 2), np.float32), np.empty((0, 8), np.float32)
    times = np.arange(0.5, cfg.speed_horizon_s + 1e-6, 1.0 / cfg.speed_hz)
    stride = max(1, int(round(cfg.window_stride_s / cfg.motion_resample_s)))
    geo_rows, speed_rows = [], []
    raw_yaw = np.asarray([x["ego"]["transform"][5] for x in kin], dtype=np.float64)
    yaw_axis = np.rad2deg(np.unwrap(np.deg2rad(raw_yaw)))
    yaw_grid = np.interp(t, np.asarray([x["sim_time"] for x in kin]), yaw_axis)
    for i in range(0, len(t), stride):
        future_t = t[i] + times
        if future_t[-1] > t[-1] + 1e-8:
            break
        j = np.searchsorted(t, future_t, side="right").clip(1, len(t) - 1)
        alpha = (future_t - t[j - 1]) / np.maximum(t[j] - t[j - 1], 1e-12)
        future = pos[j - 1] + (pos[j] - pos[j - 1]) * alpha[:, None]
        future_geo = pos[i + 1:]
        if len(future_geo) < 2:
            continue
        raw_steps = np.linalg.norm(np.diff(np.vstack((pos[i:i + 1], future_geo)), axis=0), axis=1)
        if np.any(raw_steps > cfg.max_motion_step_m):
            continue
        current_pose = np.array([pos[i, 0], pos[i, 1], pos[i, 2], 0.0, 0.0, yaw_grid[i]])
        local = transform_points(future_geo, world_to_ego(current_pose))[:, :2]
        local = np.vstack((np.zeros((1, 2)), local))
        arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(local, axis=0), axis=1))))
        # 采集数据没有倒车；过滤重采样/姿态异常造成的原点后伪倒退轨迹。
        if local[1, 0] < 0.5 or abs(local[1, 1]) > 1.5 * local[1, 0]:
            continue
        fit_horizon = cfg.geometry_horizon_m + (cfg.spline_fit_extra_m if cfg.spline_enabled else 0.0)
        if arc[-1] < fit_horizon:
            continue
        distances = np.arange(cfg.geometry_interval_m, fit_horizon + 1e-6,
                              cfg.geometry_interval_m)
        sampled = np.column_stack([np.interp(distances, arc, local[:, k]) for k in range(2)])
        vectors = np.diff(np.vstack((np.zeros((1, 2)), sampled)), axis=0)
        norms = np.linalg.norm(vectors, axis=1)
        unit = vectors / np.maximum(norms[:, None], 1e-6)
        turns = np.degrees(np.arccos(np.clip(np.sum(unit[1:] * unit[:-1], axis=1), -1.0, 1.0)))
        if np.any(turns > cfg.max_geometry_heading_change_deg):
            continue
        geo_rows.append(sampled)
        speed_rows.append(speed[np.searchsorted(t, future_t).clip(0, len(speed) - 1)])
    return np.asarray(geo_rows, dtype=np.float32), np.asarray(speed_rows, dtype=np.float32)


def _collect(cfg):
    geos, speeds, scenes, total = [], [], 0, 0
    root = Path(cfg.scene_root)
    scene_list = [root] if (root / "lmdb").is_dir() else list_scenes(root)
    for scene in scene_list:
        try:
            reader = SceneReader(scene)
            if reader.failed or bool(reader.meta.get("complete", True)) is False:
                reader.close(); continue
            g, s = _window_features(reader.kinematics(), cfg)
            reader.close()
        except Exception:
            continue
        if len(g):
            geos.append(g); speeds.append(s); scenes += 1; total += len(g)
            if total >= cfg.max_samples:
                break
    if not geos:
        raise RuntimeError("没有找到同时满足 50m/4s 的有效 ego 窗口")
    geometry, speed = np.concatenate(geos), np.concatenate(speeds)
    if len(geometry) > cfg.max_samples:
        rng = np.random.default_rng(cfg.random_seed)
        idx = rng.choice(len(geometry), cfg.max_samples, replace=False); idx.sort()
        geometry, speed = geometry[idx], speed[idx]
    return geometry, speed, scenes, total


def _fts(x, k, block, seed):
    rng = np.random.default_rng(seed); centers = np.empty((k, x.shape[1]), np.float32)
    chosen = int(rng.integers(len(x))); centers[0] = x[chosen]
    nearest = np.full(len(x), np.inf, np.float32)
    for c in range(1, k):
        for a in range(0, len(x), block):
            b = min(a + block, len(x)); d = ((x[a:b] - centers[c - 1]) ** 2).sum(1)
            nearest[a:b] = np.minimum(nearest[a:b], d)
        chosen = int(np.argmax(nearest)); centers[c] = x[chosen]
    return centers


def _kmeans(x, k, cfg):
    try:
        from sklearn.cluster import MiniBatchKMeans
        return MiniBatchKMeans(n_clusters=k, batch_size=cfg.kmeans_batch_size,
                               max_iter=cfg.kmeans_iterations, random_state=cfg.random_seed,
                               n_init=1).fit(x).cluster_centers_.astype(np.float32)
    except ImportError:
        centers = _fts(x, k, cfg.sample_block_size, cfg.random_seed)
        for _ in range(min(cfg.kmeans_iterations, 20)):
            labels = np.argmin(((x[:, None] - centers[None]) ** 2).sum(2), 1)
            for j in range(k):
                if np.any(labels == j): centers[j] = x[labels == j].mean(0)
        return centers


def _evaluate(x, centers, block):
    nearest = np.full(len(x), np.inf, np.float32); counts = np.zeros(len(centers), np.int64)
    for a in range(0, len(x), block):
        b = min(a + block, len(x)); d = ((x[a:b, None] - centers[None]) ** 2).sum(2)
        nearest[a:b] = np.sqrt(d.min(1)); counts += np.bincount(d.argmin(1), minlength=len(centers))
    pair = np.sqrt(((centers[:, None] - centers[None]) ** 2).sum(2)); pair[pair == 0] = np.inf
    return {"mean_nearest_distance": float(nearest.mean()), "p95_nearest_distance": float(np.percentile(nearest, 95)),
            "coverage_ratio": float(np.mean(nearest <= np.percentile(nearest, 95))),
            "min_center_distance": float(pair.min()), "mean_center_nearest_distance": float(pair.min(1).mean()),
            "counts": counts}


def _medoids(x, centers, block):
    """把 K-Means 均值中心投影到真实样本，避免平均曲线产生不存在的几何形状。"""
    chosen = np.empty(len(centers), dtype=np.int64)
    best = np.full(len(centers), np.inf, dtype=np.float32)
    for a in range(0, len(x), block):
        b = min(a + block, len(x))
        d = ((x[a:b, None] - centers[None]) ** 2).sum(2)
        labels = d.argmin(1)
        for j in range(len(centers)):
            hit = np.flatnonzero(labels == j)
            if len(hit):
                local = d[hit, j]; q = int(hit[local.argmin()]); value = float(local.min())
                if value < best[j]:
                    best[j] = value; chosen[j] = a + q
    return x[chosen]


def _smooth_geometry(vocab, cfg):
    """对已入选真实轨迹做参数样条平滑，再按弧长回采样，避免跨样本生成伪轨迹。"""
    if not cfg.spline_enabled:
        return vocab
    try:
        from scipy.interpolate import UnivariateSpline
    except ImportError:
        return vocab
    target = np.arange(1, vocab.shape[1] + 1, dtype=np.float64) * cfg.geometry_interval_m
    dense = np.arange(0.0, cfg.geometry_horizon_m + cfg.spline_dense_step_m * 0.5,
                      cfg.spline_dense_step_m)
    out = np.empty_like(vocab)
    base = np.arange(vocab.shape[1], dtype=np.float64) * cfg.geometry_interval_m
    for i, curve in enumerate(vocab):
        splines = [UnivariateSpline(base, curve[:, j], s=cfg.spline_smoothing, k=min(3, len(curve) - 1))
                   for j in range(2)]
        smooth = np.column_stack([spline(dense) for spline in splines])
        smooth[0] = 0.0; smooth[-1] = curve[-1]
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(smooth, axis=0), axis=1))]
        out[i] = np.column_stack([np.interp(target, arc, smooth[:, j]) for j in range(2)])
    return out


def _build(x, requested, cfg, kind):
    flat = x.reshape(len(x), -1); k = _size(requested, cfg, len(flat))
    centers = _fts(flat, k, cfg.sample_block_size, cfg.random_seed) if cfg.algorithm == "fts" else _kmeans(flat, k, cfg)
    if cfg.algorithm == "kmeans":
        centers = _medoids(flat, centers, cfg.sample_block_size) if kind == "geometry" else centers
    if kind == "geometry":
        centers = _smooth_geometry(centers.reshape((k,) + x.shape[1:]), cfg).reshape(k, -1)
        target_n = int(round(cfg.geometry_horizon_m / cfg.geometry_interval_m))
        centers = centers.reshape((k,) + x.shape[1:])[:, :target_n]
        flat = flat.reshape((len(flat),) + x.shape[1:])[:, :target_n].reshape(len(flat), -1)
        return centers, _evaluate(flat, centers.reshape(k, -1), cfg.sample_block_size)
    if kind == "speed":
        centers[0] = 0.0
    metrics = _evaluate(flat, centers, cfg.sample_block_size)
    return centers.reshape((k,) + x.shape[1:]), metrics


def generate_vocabulary(cfg):
    """扫描全量场景并生成几何、速度词表及 PNG 所需统计。"""
    start = time.perf_counter(); geometry, speed, scenes, total = _collect(cfg)
    gv, gm = _build(geometry, cfg.geometry_size, cfg, "geometry")
    sv, sm = _build(speed, cfg.speed_size, cfg, "speed")
    out = Path(cfg.output_dir); out.mkdir(parents=True, exist_ok=True)
    base = {"algorithm": cfg.algorithm, "sample_count": len(geometry), "scene_count": scenes,
            "source_window_count": total, "elapsed_s": time.perf_counter() - start, "version": 1}
    np.save(out / cfg.geometry_output, {**base, "kind": "geometry", "vocab": gv, "metrics": gm}, allow_pickle=True)
    np.save(out / cfg.speed_output, {**base, "kind": "speed", "vocab": sv, "metrics": sm}, allow_pickle=True)
    return {"geometry": gv, "speed": sv, "metrics": {"geometry": gm, "speed": sm}, **base}
