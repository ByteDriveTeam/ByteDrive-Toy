# 本文件为 train/losses/losses.py 的校验伴随文件（规范 §7.1，免文件头）。

_REQUIRED_OUTPUTS = ("semantic", "depth")
_REQUIRED_TARGETS = ("semantic", "depth_target", "depth_inrange")

_DRIVING_OUTPUTS = (
    "risk", "drivable", "distribution", "lane_class_logits", "lane_direction",
    "trajectories", "confidence", "behavior_logits",
)
_DRIVING_TARGETS = (
    "risk", "drivable", "offroad_distance", "distribution", "lane_class", "lane_direction", "inview",
    "trajectory", "traj_valid", "sector", "behavior",
)


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
    """校验对象: compute_driving_losses 入参 —— 三场、道路线图、轨迹/行为与地图监督齐备。"""
    missing_o = [k for k in _DRIVING_OUTPUTS if k not in outputs]
    missing_t = [k for k in _DRIVING_TARGETS if k not in targets]
    if missing_o:
        raise KeyError("driving outputs 缺少键: {}".format(missing_o))
    if missing_t:
        raise KeyError("driving targets 缺少键: {}".format(missing_t))
    if outputs["behavior_logits"].shape != targets["behavior"].shape:
        raise ValueError("behavior_logits 与 behavior 标签形状须一致，实际 {} / {}。".format(
            tuple(outputs["behavior_logits"].shape), tuple(targets["behavior"].shape)))
