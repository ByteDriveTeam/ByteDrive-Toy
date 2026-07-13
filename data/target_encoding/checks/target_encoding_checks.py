# 本文件为 data/target_encoding/target_encoding.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_map_2d(x, name):
    """校验对象: 深度/单通道图入参 —— 至少 2 维（末两维为 H, W）。"""
    if x.ndim < 2:
        raise ValueError("{} 期望末两维为 [H,W]，实际 ndim={}。".format(name, x.ndim))
