# 本文件为 data/target_encoding.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_map_2d(x, name):
    """校验对象: 深度/单通道图入参 —— 至少 2 维（末两维为 H, W）。"""
    if x.ndim < 2:
        raise ValueError("{} 期望末两维为 [H,W]，实际 ndim={}。".format(name, x.ndim))


def check_flow_inputs(flow, depth_m):
    """校验对象: flow_velocity_targets 入参 —— 光流末维为 2、且 H/W 与深度一致。"""
    if flow.ndim < 3 or int(flow.shape[-1]) != 2:
        raise ValueError("flow 期望 [...,H,W,2]，实际 shape={}。".format(tuple(flow.shape)))
    if flow.shape[:-1] != depth_m.shape:
        raise ValueError(
            "flow 的 [...,H,W] {} 必须与 depth_m {} 一致。".format(
                tuple(flow.shape[:-1]), tuple(depth_m.shape))
        )
