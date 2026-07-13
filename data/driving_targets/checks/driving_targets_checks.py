# 本文件为 data/driving_targets/driving_targets.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_bev_params(bev):
    """校验对象: BevParams —— 量程 min<max、分辨率为正。"""
    assert bev.x_min < bev.x_max and bev.y_min < bev.y_max, "BevParams 需 x_min<x_max 且 y_min<y_max"
    assert bev.height > 0 and bev.width > 0, "BevParams.height/width 必须 > 0"


def check_depth_map(depth_m):
    """校验对象: risk_field 入参 depth_m —— 二维深度图 (Hc, Wc)。"""
    if depth_m.ndim != 2:
        raise ValueError("depth_m 期望二维 (Hc,Wc)，实际 {}。".format(tuple(depth_m.shape)))
