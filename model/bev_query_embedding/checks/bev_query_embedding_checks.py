# 本文件为 model/bev_query_embedding/bev_query_embedding.py 的校验伴随文件（规范 §7.1，免文件头）。

import torch


def check_bev_query_args(out_dim, height, width, mlp_hidden, coord_symlog_scale):
    """校验对象: BevQueryEmbedding 构造入参 —— 维度/分辨率/隐藏维为正且尺度为有限正数。"""
    if out_dim < 1 or height < 1 or width < 1 or mlp_hidden < 1:
        raise ValueError("out_dim/height/width/mlp_hidden 必须为正整数，实际 {}/{}/{}/{}。".format(
            out_dim, height, width, mlp_hidden))
    if not torch.isfinite(torch.tensor(coord_symlog_scale)) or coord_symlog_scale <= 0:
        raise ValueError("coord_symlog_scale 必须为有限正数，实际为 {}。".format(coord_symlog_scale))


def check_bev_query_inputs(batch_size, device, grid_xy, height, width):
    """校验对象: BevQueryEmbedding.forward 入参 —— 批量、设备及可选实际网格须合法。"""
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size 必须为正整数，实际为 {}。".format(batch_size))
    if not isinstance(device, torch.device):
        raise TypeError("device 必须为 torch.device，实际为 {}。".format(type(device).__name__))
    if grid_xy is None:
        return
    expected = (batch_size, height, width, 2)
    if not torch.is_floating_point(grid_xy) or tuple(grid_xy.shape) != expected:
        raise ValueError("grid_xy 期望浮点形状 {}，实际 {} / {}。".format(
            expected, tuple(grid_xy.shape), grid_xy.dtype))
