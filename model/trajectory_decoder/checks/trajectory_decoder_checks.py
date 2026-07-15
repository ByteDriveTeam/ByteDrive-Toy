# 本文件为 model/trajectory_decoder/trajectory_decoder.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_trajectory_inputs(bev_feat, work_dim):
    """校验对象: TrajectoryDecoder.forward 入参 —— BEV 特征须为 [B,work_dim,Hb,Wb]。"""
    if bev_feat.ndim != 4:
        raise ValueError("bev_feat 期望 (B,C,Hb,Wb) 四维，实际 {}。".format(tuple(bev_feat.shape)))
    if int(bev_feat.shape[1]) != work_dim:
        raise ValueError("bev_feat 通道应为 work_dim={}，实际 {}。".format(work_dim, int(bev_feat.shape[1])))
