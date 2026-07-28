# 本文件为 model/bev_decoder/bev_decoder.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_bev_features(bev_feat, work_dim):
    """校验对象: BevDecoder.forward 入参 bev_feat —— 须为通道匹配的四维 BEV 特征。"""
    if bev_feat.ndim != 4 or int(bev_feat.shape[1]) != work_dim:
        raise ValueError("bev_feat 期望 [B,{},H,W]，实际 {}。".format(
            work_dim, tuple(bev_feat.shape)))
