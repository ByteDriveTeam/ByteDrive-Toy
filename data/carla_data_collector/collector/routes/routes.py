"""由可达点构建路线队列：距离过滤、随机排序，并剔除起终点邻近的相似路线。

模块: collector/routes/routes.py
依赖: itertools, numpy, collector.routes.checks.routes_checks
读取配置: 由 build_route_queue 接收 route.min_distance_m/max_distance_m/similarity_threshold_m/
          queue_seed 及当前地图的场景数限制，自身不读 config
对外接口:
    - build_route_queue(spawn_points, min_d, max_d, seed, max_scenes, similarity_threshold=None,
                        excluded_pairs=None) -> list[dict]
        每项: {"start_idx","end_idx","start"(pose6),"end"(pose6)}
说明: Design ③。候选有序对先按 queue_seed 固定排序；若两条同向路线的起点与终点平面距离
      分别不超过 similarity_threshold，则仅保留排序靠前者。反向路线仍视为不同路线。
"""

from itertools import chain, product

import numpy as np

from collector.routes.checks.routes_checks import check_spawn_points


__all__ = ["build_route_queue"]

_NEIGHBOR_CELL_OFFSETS = np.asarray(tuple(product((-1, 0, 1), repeat=4)), dtype=np.int8)


def _filter_similar_pairs(starts, ends, coords, threshold, excluded_pairs):
    if not threshold or starts.size == 0:
        return starts, ends

    valid_excluded = sorted(
        (start, end) for start, end in excluded_pairs
        if 0 <= start < coords.shape[0] and 0 <= end < coords.shape[0]
    )
    excluded_count = len(valid_excluded)
    if excluded_count:
        excluded = np.asarray(valid_excluded, dtype=np.int64)
        starts = np.concatenate((excluded[:, 0], starts))
        ends = np.concatenate((excluded[:, 1], ends))

    route_xy = np.concatenate((coords[starts, :2], coords[ends, :2]), axis=1)
    cells = np.floor(route_xy / threshold).astype(np.int64)
    threshold_sq = threshold * threshold
    buckets = {}
    kept = []

    for candidate, cell in enumerate(cells):
        if candidate >= excluded_count:
            nearby = np.fromiter(chain.from_iterable(
                buckets.get(tuple(cell + offset), ()) for offset in _NEIGHBOR_CELL_OFFSETS
            ), dtype=np.int64)
            if nearby.size:
                start_delta = coords[starts[nearby], :2] - coords[starts[candidate], :2]
                end_delta = coords[ends[nearby], :2] - coords[ends[candidate], :2]
                similar = (np.einsum("ij,ij->i", start_delta, start_delta) <= threshold_sq) & (
                    np.einsum("ij,ij->i", end_delta, end_delta) <= threshold_sq
                )
                if np.any(similar):
                    continue
        buckets.setdefault(tuple(cell), []).append(candidate)
        if candidate >= excluded_count:
            kept.append(candidate)

    kept = np.asarray(kept, dtype=np.int64)
    return starts[kept], ends[kept]


def build_route_queue(spawn_points, min_d, max_d, seed, max_scenes,
                      similarity_threshold=None, excluded_pairs=None):
    """构建剔除相似项且按 seed 随机排序的路线队列。

    相似判定只比较同向路线：起点平面距离与终点平面距离均不超过阈值时，保留随机排序靠前者。
    阈值为 0 或未传时关闭相似路线剔除；excluded_pairs 中的已采路线及其相似路线优先排除。
    """
    check_spawn_points(spawn_points)
    excluded_pairs = set(excluded_pairs or ())

    coords = np.array([p[:3] for p in spawn_points], dtype=np.float64)  # (N,3)
    # 向量化两两欧氏距离：N×N 距离矩阵
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)

    mask = (dist >= min_d) & (dist <= max_d)
    np.fill_diagonal(mask, False)  # 排除自身到自身
    starts, ends = np.nonzero(mask)

    rng = np.random.RandomState(seed)
    order = rng.permutation(starts.shape[0])  # 随机排序但可复现
    starts, ends = starts[order], ends[order]
    if excluded_pairs:
        keep = np.fromiter(
            ((int(start), int(end)) not in excluded_pairs for start, end in zip(starts, ends)),
            dtype=bool,
        )
        starts, ends = starts[keep], ends[keep]
    starts, ends = _filter_similar_pairs(
        starts, ends, coords, similarity_threshold, excluded_pairs
    )
    if max_scenes > 0:
        starts, ends = starts[:max_scenes], ends[:max_scenes]

    return [{"start_idx": int(i), "end_idx": int(j),
             "start": spawn_points[i], "end": spawn_points[j]}
            for i, j in zip(starts, ends)]
