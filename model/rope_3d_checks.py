# 本文件为 model/rope_3d.py 的校验伴随文件（规范 §7.1，免文件头）。

import torch


def check_axis_dims(axis_dims):
    """校验对象: apply_rope_3d / RoPE3D 的 axis_dims —— 3 个正偶数；返回校验后的三元组。"""
    try:
        raw_axis_dims = tuple(axis_dims)
    except TypeError as exc:
        raise TypeError(
            "axis_dims 必须是包含 3 个整数的序列，实际为 {!r}。".format(axis_dims)
        ) from exc
    if len(raw_axis_dims) != 3:
        raise ValueError("axis_dims 必须包含 3 个整数，实际为 {}。".format(raw_axis_dims))

    validated_dims = []
    for axis_index, raw_axis_dim in enumerate(raw_axis_dims):
        if not isinstance(raw_axis_dim, int) or isinstance(raw_axis_dim, bool):
            raise TypeError("axis_dims[{}] 必须为整数，实际为 {!r}。".format(axis_index, raw_axis_dim))
        if raw_axis_dim <= 0:
            raise ValueError("axis_dims[{}] 必须为正偶数，实际为 {}。".format(axis_index, raw_axis_dim))
        if raw_axis_dim % 2 != 0:
            raise ValueError("axis_dims[{}] 必须为偶数，实际为 {}。".format(axis_index, raw_axis_dim))
        validated_dims.append(int(raw_axis_dim))
    return tuple(validated_dims)


def check_theta(theta):
    """校验对象: apply_rope_3d / RoPE3D 的 theta —— 必须为正数值。"""
    if not isinstance(theta, (int, float)) or isinstance(theta, bool):
        raise TypeError("theta 必须为数值，实际为 {}。".format(type(theta).__name__))
    if float(theta) <= 0.0:
        raise ValueError("theta 必须为正数，实际为 {}。".format(theta))


def check_rope_inputs(features, positions):
    """校验对象: apply_rope_3d 入参 features/positions —— 浮点、同设备、shape 与 token 数匹配。"""
    if not torch.is_floating_point(features):
        raise TypeError("features 必须为浮点张量，实际 dtype 为 {}。".format(features.dtype))
    if not torch.is_floating_point(positions):
        raise TypeError("positions 必须为浮点张量，实际 dtype 为 {}。".format(positions.dtype))
    if features.device != positions.device:
        raise ValueError(
            "features 和 positions 必须位于同一设备，实际分别为 {} 和 {}。".format(
                features.device, positions.device
            )
        )
    if features.ndim < 2:
        raise ValueError(
            "features 期望 shape 为 [..., N, C]，实际 shape 为 {}。".format(tuple(features.shape))
        )
    if positions.ndim < 2 or int(positions.shape[-1]) != 3:
        raise ValueError(
            "positions 期望 shape 为 [N, 3] 或 [..., N, 3]，实际为 {}。".format(tuple(positions.shape))
        )
    if int(features.shape[-2]) != int(positions.shape[-2]):
        raise ValueError(
            "features 和 positions 的 token 数必须一致，实际分别为 {} 和 {}。".format(
                features.shape[-2], positions.shape[-2]
            )
        )


def check_rope_capacity(rotary_dim, feature_dim):
    """校验对象: axis_dims 总和 —— 不能超过 features 最后一维通道数。"""
    if rotary_dim > feature_dim:
        raise ValueError(
            "axis_dims 总和不能超过 features 最后一维，实际为 {} > {}。".format(rotary_dim, feature_dim)
        )


def check_query_key_match(query, key):
    """校验对象: RoPE3D.forward 入参 query/key —— shape 必须一致。"""
    if query.shape != key.shape:
        raise ValueError(
            "query 和 key 的 shape 必须一致，实际为 {} 和 {}。".format(
                tuple(query.shape), tuple(key.shape)
            )
        )
