# 本文件为 train/losses/losses.py 的校验伴随文件（规范 §7.1，免文件头）。

_REQUIRED_OUTPUTS = ("semantic", "depth")
_REQUIRED_TARGETS = ("semantic", "depth_target", "depth_inrange")

_DRIVING_OUTPUTS = ("risk", "drivable", "distribution", "trajectories", "confidence")
_DRIVING_TARGETS = ("risk", "drivable", "distribution", "inview", "trajectory", "traj_valid", "sector")


def check_losses_io(outputs, targets):
    """校验对象: compute_losses 入参 —— 双头输出与三目标齐备且深度头有 2 通道。"""
    missing_o = [k for k in _REQUIRED_OUTPUTS if k not in outputs]
    missing_t = [k for k in _REQUIRED_TARGETS if k not in targets]
    if missing_o:
        raise KeyError("outputs 缺少键: {}".format(missing_o))
    if missing_t:
        raise KeyError("targets 缺少键: {}".format(missing_t))
    if int(outputs["depth"].shape[1]) < 2:
        raise ValueError("depth 头须至少 2 通道（回归+范围二分类），实际 {}。".format(
            int(outputs["depth"].shape[1])))


def check_driving_losses_io(outputs, targets):
    """校验对象: compute_driving_losses 入参 —— 三场+轨迹+置信度输出与对应 GT 齐备。"""
    missing_o = [k for k in _DRIVING_OUTPUTS if k not in outputs]
    missing_t = [k for k in _DRIVING_TARGETS if k not in targets]
    if missing_o:
        raise KeyError("driving outputs 缺少键: {}".format(missing_o))
    if missing_t:
        raise KeyError("driving targets 缺少键: {}".format(missing_t))
