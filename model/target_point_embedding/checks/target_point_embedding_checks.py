# 本文件为 model/target_point_embedding/target_point_embedding.py 的校验伴随文件（规范 §7.1，免文件头）。

import torch


def check_bev_query_args(out_dim, height, width, mlp_hidden, coord_symlog_scale, vector_order):
    """校验对象: TargetPointEmbedding 构造入参 —— 维度/分辨率/隐藏维为正、尺度为正、向量方向枚举合法。"""
    if out_dim < 1 or height < 1 or width < 1 or mlp_hidden < 1:
        raise ValueError("out_dim/height/width/mlp_hidden 必须为正整数，实际 {}/{}/{}/{}。".format(
            out_dim, height, width, mlp_hidden))
    if coord_symlog_scale <= 0:
        raise ValueError("coord_symlog_scale 必须为正数，实际为 {}。".format(coord_symlog_scale))
    if vector_order not in ("grid_minus_target", "target_minus_grid"):
        raise ValueError("vector_order 仅支持 grid_minus_target / target_minus_grid，实际 {}。".format(vector_order))


def check_target_points(target_points):
    """校验对象: TargetPointEmbedding.forward 入参 target_points —— 浮点且 shape 为 [B, 2]（ego 平面 [x,y]）。"""
    if not torch.is_floating_point(target_points):
        raise TypeError("target_points 必须为浮点张量，实际 dtype 为 {}。".format(target_points.dtype))
    if target_points.ndim != 2 or int(target_points.shape[1]) != 2:
        raise ValueError("target_points 期望 [B, 2]（ego [x,y]），实际为 {}。".format(tuple(target_points.shape)))
