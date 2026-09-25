# 本文件为 model/trajectory_decoder/trajectory_decoder.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_trajectory_inputs(features, detect_anchors, target_point, ego_velocity,
                            work_dim, num_bev, trajectory):
    """校验对象: TrajectoryDecoder.forward —— 六层完整查询、锚点与条件须逐批对齐。"""
    if len(features) != 6:
        raise ValueError("perception_features 须恰含六层完整感知查询，实际 {} 路。".format(len(features)))
    if target_point.ndim != 2 or int(target_point.shape[1]) != 2:
        raise ValueError("target_point 期望 [B,2]，实际 {}。".format(tuple(target_point.shape)))
    if ego_velocity.shape != target_point.shape:
        raise ValueError("ego_velocity 须与 target_point 同为 [B,2]，实际 {} / {}。".format(
            tuple(ego_velocity.shape), tuple(target_point.shape)))
    batch_size = int(target_point.shape[0])
    if detect_anchors.ndim != 3 or detect_anchors.shape[0] != batch_size \
            or detect_anchors.shape[2] != 3:
        raise ValueError("detect_anchors 须为 [B,N_detect,3]")
    num_tokens = num_bev + int(detect_anchors.shape[1])
    if any(feature.ndim != 3 or tuple(feature.shape) !=
           (batch_size, num_tokens, work_dim) for feature in features):
        raise ValueError("六层感知特征均须为 [B,N_bev+N_detect,work_dim]")
    if trajectory is not None and tuple(trajectory.shape) != (batch_size, 12, 2):
        raise ValueError("trajectory 须为 [B,12,2]")
