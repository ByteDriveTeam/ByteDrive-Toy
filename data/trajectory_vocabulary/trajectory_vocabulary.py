"""从 CARLA ego 时间轴生成几何、速度和局部二维加速度词表。

模块: data/trajectory_vocabulary/trajectory_vocabulary.py
依赖: numpy, data.single_frame_base, vis.data_vis.reader
读取配置: trajectory_vocabulary.*、model.unified_driving.nearest_weight_floor/nearest_decay_s
对外接口:
    - generate_vocabulary(cfg) -> dict
    - generate_action_vocabulary(cfg, model_spec) -> dict
说明: 先按 0.5 秒重采样，再以块状计算特征和距离，避免保存全量高频时间轴及距离矩阵。
"""

from pathlib import Path
import time

import numpy as np
import torch

from data.driving_cache import trajectory_to_planar_accelerations
from data.single_frame_base import successful_scene_meta
from model.driving_model.unified_driving_model import signed_log
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
            reader = SceneReader(scene, open_videos=False)
            if not successful_scene_meta(reader.meta):
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


def _fts_indices(x, k, block, seed, first=None):
    """按最远轨迹采样返回真实窗口索引；相同特征不会重复占词。"""
    rng = np.random.default_rng(seed)
    selected = np.empty(k, dtype=np.int64)
    selected[0] = int(rng.integers(len(x))) if first is None else int(first)
    nearest = np.full(len(x), np.inf, np.float32)
    for index in range(1, k):
        for a in range(0, len(x), block):
            b = min(a + block, len(x)); d = ((x[a:b] - x[selected[index - 1]]) ** 2).sum(1)
            nearest[a:b] = np.minimum(nearest[a:b], d)
        selected[index] = int(np.argmax(nearest))
        if nearest[selected[index]] <= 0:
            raise ValueError("独立轨迹不足以生成指定数量的最远采样词")
    return selected


def _fts(x, k, block, seed):
    return x[_fts_indices(x, k, block, seed)]


def _kmeans(x, k, cfg):
    try:
        from sklearn.cluster import MiniBatchKMeans
        return MiniBatchKMeans(n_clusters=k, batch_size=cfg.kmeans_batch_size,
                               max_iter=cfg.kmeans_iterations, random_state=cfg.random_seed,
                               n_init=1).fit(x).cluster_centers_.astype(np.float32)
    except ImportError:
        rng = np.random.default_rng(cfg.random_seed)
        subset = x[rng.choice(len(x), size=min(len(x), cfg.sample_block_size), replace=False)]
        centers = np.empty((k, x.shape[1]), dtype=np.float32)
        centers[0] = subset[rng.integers(len(subset))]
        nearest = np.full(len(subset), np.inf, dtype=np.float64)
        for index in range(1, k):
            nearest = np.minimum(nearest, ((subset - centers[index - 1]) ** 2).sum(-1))
            probability = nearest / nearest.sum() if nearest.sum() > 0 else None
            centers[index] = subset[rng.choice(len(subset), p=probability)]
        counts = np.zeros(k, dtype=np.float64)
        for _ in range(cfg.kmeans_iterations):
            batch = x[rng.integers(len(x), size=cfg.kmeans_batch_size)]
            distance = ((batch * batch).sum(-1, keepdims=True)
                        + (centers * centers).sum(-1)[None] - 2.0 * batch @ centers.T)
            labels = distance.argmin(-1)
            observed = np.bincount(labels, minlength=k)
            sums = np.zeros_like(centers)
            np.add.at(sums, labels, batch)
            active = observed > 0
            centers[active] = ((counts[active, None] * centers[active] + sums[active])
                               / (counts[active, None] + observed[active, None]))
            counts += observed
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


def _action_windows(kinematics, cfg):
    """按动作采样频率取得真实未来轨迹、物理动作与当前运动学状态。"""
    if len(kinematics) < 3:
        return None
    raw_time = np.asarray([row["sim_time"] for row in kinematics], dtype=np.float64)
    valid = np.r_[True, np.diff(raw_time) > 0]
    raw_time = raw_time[valid]
    rows = [row for row, keep in zip(kinematics, valid) if keep]
    if len(raw_time) < 3:
        return None
    step = 1.0 / cfg.action_hz
    points = int(round(cfg.action_horizon_s * cfg.action_hz))
    grid = np.arange(raw_time[0], raw_time[-1] + step * 0.25, step)
    grid = grid[grid <= raw_time[-1] + 1e-8]
    if len(grid) <= points:
        return None
    pose = np.asarray([row["ego"]["transform"] for row in rows], dtype=np.float64)
    velocity = np.asarray([row["ego"]["velocity"][:2] for row in rows], dtype=np.float64)
    acceleration = np.asarray([row["ego"].get("acceleration", [0, 0])[:2]
                               for row in rows], dtype=np.float64)
    xy = np.stack([np.interp(grid, raw_time, pose[:, axis]) for axis in (0, 1)], axis=-1)
    yaw = np.interp(grid, raw_time, np.unwrap(np.deg2rad(pose[:, 5])))
    motion = np.stack([np.interp(grid, raw_time, column)
                       for column in np.concatenate((velocity, acceleration), axis=1).T], axis=-1)
    starts = np.arange(0, len(grid) - points,
                       max(1, int(round(cfg.window_stride_s * cfg.action_hz))))
    future = xy[starts[:, None] + np.arange(1, points + 1)]
    delta = future - xy[starts, None]
    cosine, sine = np.cos(yaw[starts])[:, None], np.sin(yaw[starts])[:, None]
    local = np.stack((cosine * delta[..., 0] + sine * delta[..., 1],
                      -sine * delta[..., 0] + cosine * delta[..., 1]), axis=-1)
    good = np.isfinite(local).all((1, 2))
    local, motion = local[good], motion[starts[good]]
    if len(local) == 0:
        return None
    motion = motion.reshape(-1, 2, 2)
    motion = np.stack((cosine[good] * motion[..., 0] + sine[good] * motion[..., 1],
                       -sine[good] * motion[..., 0] + cosine[good] * motion[..., 1]),
                      axis=-1).reshape(-1, 4)
    actions = trajectory_to_planar_accelerations(
        torch.from_numpy(local).float(), torch.from_numpy(motion).float(), step).numpy()
    return (local.astype(np.float32), actions.astype(np.float32),
            motion[:, :2].astype(np.float32), motion.astype(np.float32))


def _action_examples(cfg):
    """只采有效场景，避免把失败行驶和缺失未来时间窗口作为词。"""
    root = Path(cfg.scene_root)
    scenes = [root] if (root / "lmdb").is_dir() else list_scenes(root)
    local_rows, action_rows, speed_rows, motion_rows = [], [], [], []
    total, used = 0, 0
    for scene in scenes:
        try:
            reader = SceneReader(scene, open_videos=False)
            if not successful_scene_meta(reader.meta):
                reader.close()
                continue
            window = _action_windows(reader.kinematics(), cfg)
            reader.close()
        except (OSError, ValueError, KeyError):
            continue
        if window is None:
            continue
        local, actions, speed, motion = window
        local_rows.append(local)
        action_rows.append(actions)
        speed_rows.append(speed)
        motion_rows.append(motion)
        total += len(actions)
        used += 1
    if total < cfg.action_size:
        raise RuntimeError("有效动作窗口少于请求的动作词数")
    indices = np.random.default_rng(cfg.random_seed).choice(
        total, size=min(total, cfg.max_samples), replace=False)
    arrays = [np.concatenate(rows)[indices]
              for rows in (local_rows, action_rows, speed_rows, motion_rows)]
    return (*arrays, used, total)


def generate_action_vocabulary(cfg, model_spec):
    """按未来轨迹最远采样生成物理动作词及独立的标准化统计。"""
    start = time.perf_counter()
    local, actions, speed, motion, scenes, count = _action_examples(cfg)
    action_mean = actions.mean(axis=(0, 1)).astype(np.float32)
    action_std = actions.std(axis=(0, 1)).clip(1e-3).astype(np.float32)
    motion_signed = signed_log(torch.from_numpy(motion)).numpy()
    motion_mean = motion_signed.mean(axis=0).astype(np.float32)
    motion_std = motion_signed.std(axis=0).clip(1e-3).astype(np.float32)
    index = np.arange(local.shape[1], dtype=np.float32)
    weights = (model_spec.nearest_weight_floor
               + (1.0 - model_spec.nearest_weight_floor)
               * np.exp(-index / (cfg.action_hz * model_spec.nearest_decay_s)))
    features = (local * np.sqrt(weights)[None, :, None]).reshape(len(local), -1)
    # 数值二阶差分会放大定位抖动；只限制成功场景中的词原型候选。
    peak_acceleration = np.linalg.norm(actions, axis=-1).max(axis=1)
    peak_jerk = np.linalg.norm(np.diff(actions, axis=1), axis=-1).max(axis=1) * cfg.action_hz
    eligible = np.flatnonzero((peak_acceleration <= cfg.action_candidate_max_accel_mps2)
                              & (peak_jerk <= cfg.action_candidate_max_jerk_mps3))
    if len(eligible) < cfg.action_size:
        raise ValueError("满足动作词候选连续性限制的窗口少于请求的词数")
    distinct = eligible[np.unique(actions[eligible].reshape(len(eligible), -1),
                                  axis=0, return_index=True)[1]]
    if len(distinct) < cfg.action_size:
        raise ValueError("独立的二维加速度窗口少于请求的词数")
    candidates = features[distinct]
    stationary = np.abs(actions[distinct]).sum((1, 2)).argmin()
    selected = distinct[_fts_indices(
        candidates, cfg.action_size, cfg.sample_block_size, cfg.random_seed, stationary)]
    words = actions[selected].astype(np.float32)
    mean = np.zeros(2, dtype=np.float64)
    square = np.zeros(2, dtype=np.float64)
    matched = 0
    chunk = max(1, cfg.sample_block_size // cfg.action_size)
    dt = 1.0 / cfg.action_hz
    for offset in range(0, len(actions), chunk):
        end = min(offset + chunk, len(actions))
        word_velocity = speed[offset:end, None, None, :] + np.cumsum(words[None], axis=2) * dt
        word_xy = np.cumsum(word_velocity, axis=2) * dt
        distance = ((word_xy - local[offset:end, None]) ** 2
                    * weights[None, None, :, None]).sum((2, 3))
        nearest = distance.argmin(axis=1)
        residual = actions[offset:end] - words[nearest]
        mean += residual.sum((0, 1))
        square += np.square(residual).sum((0, 1))
        matched += residual.shape[0] * residual.shape[1]
    residual_mean = (mean / matched).astype(np.float32)
    residual_std = np.sqrt(np.maximum(square / matched - residual_mean.astype(np.float64) ** 2,
                                      1e-6)).astype(np.float32)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    destination = out / cfg.action_output
    np.savez_compressed(destination, actions=words,
                        normalized_actions=((words - action_mean) / action_std).astype(np.float32),
                        source_trajectory=local[selected].astype(np.float32),
                        source_velocity=speed[selected].astype(np.float32),
                        candidate_count=np.int64(len(eligible)),
                        action_mean=action_mean,
                        action_std=action_std, residual_mean=residual_mean,
                        residual_std=residual_std, motion_mean=motion_mean,
                        motion_std=motion_std, horizon_s=np.float32(cfg.action_horizon_s),
                        hz=np.float32(cfg.action_hz))
    return {"path": str(destination), "scene_count": scenes,
            "sample_count": len(actions), "source_window_count": count,
            "unique_action_count": len(distinct), "candidate_count": len(eligible),
            "elapsed_s": time.perf_counter() - start}
