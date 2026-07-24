"""由 CARLA 推荐生成点构造可复现、无重复的闭环评测路线队列。

模块: clone_loop/routes/routes.py
依赖: numpy, clone_loop.routes.checks.routes_checks
读取配置: 参数由调用方从 clone_loop.route 传入
对外接口:
    - build_route_queue(spawn_points, min_distance, max_distance, seed, max_episodes) -> list[dict]
"""

import numpy as np

from clone_loop.routes.checks.routes_checks import check_route_inputs


__all__ = ["build_route_queue"]


def build_route_queue(spawn_points, min_distance, max_distance, seed, max_episodes):
    """按起终点直线距离筛选有序点对，并用固定种子打乱。"""
    check_route_inputs(spawn_points, min_distance, max_distance, max_episodes)
    poses = np.asarray(spawn_points, dtype=np.float64)
    distances = np.linalg.norm(poses[:, None, :2] - poses[None, :, :2], axis=-1)
    starts, ends = np.nonzero((distances >= min_distance) & (distances <= max_distance))
    pairs = np.stack((starts, ends), axis=1)
    np.random.RandomState(seed).shuffle(pairs)
    if max_episodes:
        pairs = pairs[:max_episodes]
    return [{
        "start_idx": int(start), "end_idx": int(end),
        "start": poses[start].tolist(), "end": poses[end].tolist(),
    } for start, end in pairs]
