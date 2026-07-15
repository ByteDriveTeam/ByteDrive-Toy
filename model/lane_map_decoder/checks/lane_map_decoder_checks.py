# 本文件为 model/lane_map_decoder/lane_map_decoder.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_lane_map_features(bev_feat, work_dim):
    """校验对象: LaneMapDecoder.forward 入参 —— 须为 [B,work_dim,Hb,Wb] 四维 BEV 特征。"""
    if bev_feat.ndim != 4 or int(bev_feat.shape[1]) != work_dim:
        raise ValueError("bev_feat 期望 (B,{},Hb,Wb)，实际 {}。".format(
            work_dim, tuple(bev_feat.shape)))
