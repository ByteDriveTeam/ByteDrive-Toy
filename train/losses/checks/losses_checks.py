# 本文件为 train/losses/losses.py 的校验伴随文件（规范 §7.1，免文件头）。

_REQUIRED_OUTPUTS = ("semantic", "depth")
_REQUIRED_TARGETS = ("semantic", "depth_target", "depth_inrange")

_DRIVING_OUTPUTS = (
    "scene_occ_logits", "agent_occ_logits", "drivable", "lane_class_logits", "lane_direction",
    "stop_line_logits", "detect_class_logits", "detect_boxes", "detect_future",
    "flow_velocity", "flow_target", "flow_valid",
)
_DRIVING_TARGETS = (
    "scene_occ", "scene_occ_mask", "agent_occ", "agent_occ_mask",
    "drivable", "lane_class", "lane_direction", "lane_direction_valid",
    "stop_line", "detect_class", "detect_box",
    "detect_future", "detect_future_valid", "trajectory", "traj_valid",
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
    """校验对象: compute_driving_losses 入参 —— 新驾驶输出与监督齐备且形状对齐。"""
    missing_o = [k for k in _DRIVING_OUTPUTS if k not in outputs]
    missing_t = [k for k in _DRIVING_TARGETS if k not in targets]
    if missing_o:
        raise KeyError("driving outputs 缺少键: {}".format(missing_o))
    if missing_t:
        raise KeyError("driving targets 缺少键: {}".format(missing_t))
    for kind in ("scene", "agent"):
        if outputs[kind + "_occ_logits"].shape != targets[kind + "_occ"].shape \
                or targets[kind + "_occ"].shape != targets[kind + "_occ_mask"].shape:
            raise ValueError("{} 占用预测、标签、掩码必须同形".format(kind))
    if outputs["flow_velocity"].shape != outputs["flow_target"].shape:
        raise ValueError("流速度预测与目标必须同形")
    lane_shape = tuple(targets["lane_class"].shape)
    if tuple(outputs["lane_class_logits"].shape[0:1] + outputs["lane_class_logits"].shape[2:]) != lane_shape \
            or tuple(targets["lane_direction_valid"].shape) != lane_shape \
            or tuple(outputs["lane_direction"].shape) != tuple(targets["lane_direction"].shape) \
            or tuple(outputs["lane_direction"].shape[0:1] + outputs["lane_direction"].shape[2:]) != lane_shape:
        raise ValueError("车道语义、方向预测、方向标签与有效掩码的空间尺寸必须一致")
