import torch


def check_lidar_fusion_inputs(
        query, visual, stats, occupied, valid, work_dim, grid_shape, bev_shape):
    """校验对象: LidarQueryFusion.forward 入参 —— BEV、视觉与体素批量形状须一致。"""
    if query.ndim != 4:
        raise ValueError("query 期望 4 维，实际 {}。".format(tuple(query.shape)))
    batch = int(query.shape[0])
    expected_query = (batch, work_dim, *bev_shape)
    if tuple(query.shape) != expected_query:
        raise ValueError("query 期望 {}，实际 {}。".format(
            expected_query, tuple(query.shape)))
    if visual.ndim != 5 or int(visual.shape[0]) != batch or int(visual.shape[2]) != work_dim:
        raise ValueError("visual 期望 [B,V,{},H,W]，实际 {}。".format(
            work_dim, tuple(visual.shape)))
    expected_stats = (batch, 6, *grid_shape)
    expected_occupied = (batch, 1, *grid_shape)
    if tuple(stats.shape) != expected_stats or stats.dtype != torch.float32:
        raise ValueError("stats 期望 FP32 {}，实际 {} / {}。".format(
            expected_stats, tuple(stats.shape), stats.dtype))
    if tuple(occupied.shape) != expected_occupied or occupied.dtype != torch.bool:
        raise ValueError("occupied 期望 bool {}，实际 {} / {}。".format(
            expected_occupied, tuple(occupied.shape), occupied.dtype))
    if valid.ndim != 1 or int(valid.shape[0]) != batch:
        raise ValueError("valid 期望 ({},)，实际 {}。".format(batch, tuple(valid.shape)))
    devices = {tensor.device for tensor in (query, visual, stats, occupied, valid)}
    if len(devices) != 1:
        raise ValueError("query/visual/stats/occupied/valid 必须位于同一设备。")
