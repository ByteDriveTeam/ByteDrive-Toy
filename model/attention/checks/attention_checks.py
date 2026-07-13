# 本文件为 model/attention/attention.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_attention_args(dim, num_heads, mlp_ratio):
    """校验对象: CrossAttentionBlock / SelfAttentionBlock 构造入参 —— 维度可被头数整除、膨胀率为正。"""
    if dim < 1:
        raise ValueError("dim 必须为正整数，实际为 {}。".format(dim))
    if num_heads < 1 or dim % num_heads != 0:
        raise ValueError("num_heads 必须为正且整除 dim，实际 dim={} num_heads={}。".format(dim, num_heads))
    if mlp_ratio < 1:
        raise ValueError("mlp_ratio 必须不小于 1，实际为 {}。".format(mlp_ratio))
    # 前馈 C→(mlp_ratio·C)→SwiGLU 折半→C：升维通道须为偶数才能二等分 value/gate
    if (dim * mlp_ratio) % 2 != 0:
        raise ValueError("dim·mlp_ratio 必须为偶数（SwiGLU 二等分），实际 dim={} mlp_ratio={}。".format(
            dim, mlp_ratio))


def check_token_features(x, dim, name):
    """校验对象: 注意力块前向入参 —— 期望 [B, N, dim] 三维、末维等于 dim。"""
    if x.ndim != 3:
        raise ValueError("{} 期望 (B,N,C) 三维，实际 {}。".format(name, tuple(x.shape)))
    if int(x.shape[-1]) != dim:
        raise ValueError("{} 末维应为 {}，实际 {}。".format(name, dim, int(x.shape[-1])))
