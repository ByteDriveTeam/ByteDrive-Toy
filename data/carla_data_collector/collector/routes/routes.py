"""由可达点构建路线队列：两两组合、按直线距离过滤、随机排序、确保不重复。

模块: collector/routes/routes.py
依赖: numpy, collector.routes.checks.routes_checks
读取配置: 由 build_route_queue 接收 route.min_distance_m/max_distance_m/queue_seed/max_scenes，自身不读 config
对外接口:
    - build_route_queue(spawn_points, min_d, max_d, seed, max_scenes) -> list[dict]
        每项: {"start_idx","end_idx","start"(pose6),"end"(pose6)}
说明: Design ③。N 个可达点构成 N×N 距离矩阵（向量化），保留 min<=d<=max 的有序对(i!=j)；
      有序对天然不重复，(i,j) 与 (j,i) 视为不同路线（起终点互换）。queue_seed 固定排序以复现队列。
"""

import numpy as np

from collector.routes.checks.routes_checks import check_spawn_points


def build_route_queue(spawn_points, min_d, max_d, seed, max_scenes):
    """构建去重且按 seed 随机排序的路线队列。"""
    check_spawn_points(spawn_points)

    coords = np.array([p[:3] for p in spawn_points], dtype=np.float64)  # (N,3)
    # 向量化两两欧氏距离：N×N 距离矩阵
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)

    mask = (dist >= min_d) & (dist <= max_d)
    np.fill_diagonal(mask, False)  # 排除自身到自身
    starts, ends = np.nonzero(mask)

    rng = np.random.RandomState(seed)
    order = rng.permutation(starts.shape[0])  # 随机排序但可复现
    starts, ends = starts[order], ends[order]
    if max_scenes > 0:
        starts, ends = starts[:max_scenes], ends[:max_scenes]

    return [{"start_idx": int(i), "end_idx": int(j),
             "start": spawn_points[i], "end": spawn_points[j]}
            for i, j in zip(starts, ends)]
