# 本文件为 model/trajectory_decoder/trajectory_decoder.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_trajectory_inputs(features, target_point, ego_velocity, work_dim, num_modes):
    """校验对象: TrajectoryDecoder.forward 入参 —— 两路感知特征与目标/速度的形状批次须一致。"""
    if len(features) != 2:
        raise ValueError("perception_features 须恰含主感知第 3、6 层，实际 {} 路。".format(len(features)))
    if target_point.ndim != 2 or int(target_point.shape[1]) != 2:
        raise ValueError("target_point 期望 [B,2]，实际 {}。".format(tuple(target_point.shape)))
    if ego_velocity.shape != target_point.shape:
        raise ValueError("ego_velocity 须与 target_point 同为 [B,2]，实际 {} / {}。".format(
            tuple(ego_velocity.shape), tuple(target_point.shape)))
    batch_size = int(target_point.shape[0])
    if any(feature.ndim != 4 or int(feature.shape[0]) != batch_size
           or int(feature.shape[1]) != work_dim for feature in features):
        raise ValueError("两路感知特征均须为 [B,work_dim,H,W]，work_dim={}、B={}。".format(
            work_dim, batch_size))
    if num_modes != 8:
        raise ValueError("规划 Mode Token 数须为 8，实际 {}。".format(num_modes))
