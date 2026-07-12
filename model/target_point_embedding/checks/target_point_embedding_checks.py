# 本文件为 model/target_point_embedding/target_point_embedding.py 的校验伴随文件（规范 §7.1，免文件头）。

import torch


def check_target_points(target_points, coordinate_dim):
    """校验对象: TargetPointEmbedding.forward 入参 target_points —— 浮点且 shape 为 [B, coordinate_dim]。"""
    if not torch.is_floating_point(target_points):
        raise TypeError("target_points 必须为浮点张量，实际 dtype 为 {}。".format(target_points.dtype))
    if target_points.ndim != 2:
        raise ValueError(
            "target_points 期望 shape 为 [B, coordinate_dim]，实际为 {}。".format(tuple(target_points.shape))
        )
    if int(target_points.shape[1]) != coordinate_dim:
        raise ValueError(
            "target_points 最后一维必须等于 coordinate_dim，期望 {}，实际为 {}。".format(
                coordinate_dim, target_points.shape[1]
            )
        )


def check_embedded_features(embedded_features, feature_channels, output_height, output_width):
    """校验对象: 目标点卷积输出 embedded_features —— 通道与空间尺寸须与配置一致。"""
    expected_shape = (feature_channels, output_height, output_width)
    actual_shape = tuple(int(dim) for dim in embedded_features.shape[1:])
    if actual_shape != expected_shape:
        raise ValueError(
            "目标点卷积输出 shape 与配置不一致：期望 [B, {}, {}, {}]，实际为 {}。".format(
                expected_shape[0], expected_shape[1], expected_shape[2], tuple(embedded_features.shape)
            )
        )
