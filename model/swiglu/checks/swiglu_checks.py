# 本文件为 model/swiglu/swiglu.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_swiglu_dim(dim):
    """校验对象: swiglu / SwiGLU 的 dim 参数 —— 必须为整数。"""
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise TypeError("dim 必须为整数，实际为 {}。".format(type(dim).__name__))


def check_swiglu_features(features, dim):
    """校验对象: swiglu 入参 features —— 至少 1 维、dim 在界内且指定维度可二等分。

    返回归一化后的非负维度索引，供实现直接用于 chunk（校验与归一化同一来源，避免重复推导）。
    """
    check_swiglu_dim(dim)
    if features.ndim == 0:
        raise ValueError("features 必须至少包含 1 个维度，实际为 0 维张量。")

    normalized_dim = dim + features.ndim if dim < 0 else dim
    if normalized_dim < 0 or normalized_dim >= features.ndim:
        raise ValueError(
            "dim 必须位于 [{}, {}]，实际为 {}。".format(-features.ndim, features.ndim - 1, dim)
        )

    split_size = int(features.shape[normalized_dim])
    if split_size % 2 != 0:
        raise ValueError(
            "features 在指定维度上的长度必须能二等分，dim={} 的实际长度为 {}。".format(dim, split_size)
        )
    return normalized_dim
