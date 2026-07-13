# 本文件为 model/trajectory_decoder/trajectory_decoder.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_trajectory_inputs(bev_feat, ego_velocity, work_dim):
    """校验对象: TrajectoryDecoder.forward 入参 —— BEV 特征 [B,work_dim,Hb,Wb]、自车速度 [B,2]，批次一致。"""
    if bev_feat.ndim != 4:
        raise ValueError("bev_feat 期望 (B,C,Hb,Wb) 四维，实际 {}。".format(tuple(bev_feat.shape)))
    if int(bev_feat.shape[1]) != work_dim:
        raise ValueError("bev_feat 通道应为 work_dim={}，实际 {}。".format(work_dim, int(bev_feat.shape[1])))
    if ego_velocity.ndim != 2 or int(ego_velocity.shape[1]) != 2:
        raise ValueError("ego_velocity 期望 [B, 2]（ego 系 vx,vy），实际 {}。".format(tuple(ego_velocity.shape)))
    if int(bev_feat.shape[0]) != int(ego_velocity.shape[0]):
        raise ValueError("bev_feat 与 ego_velocity 批次须一致，实际 {} / {}。".format(
            int(bev_feat.shape[0]), int(ego_velocity.shape[0])))
