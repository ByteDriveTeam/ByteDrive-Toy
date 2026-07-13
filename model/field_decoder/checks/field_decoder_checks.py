# 本文件为 model/field_decoder/field_decoder.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_bev_feat(bev_feat, work_dim):
    """校验对象: FieldDecoder.forward 入参 bev_feat —— [B, work_dim, Hb, Wb] 四维、通道等于 work_dim。"""
    if bev_feat.ndim != 4:
        raise ValueError("bev_feat 期望 (B,C,Hb,Wb) 四维，实际 {}。".format(tuple(bev_feat.shape)))
    if int(bev_feat.shape[1]) != work_dim:
        raise ValueError("bev_feat 通道应为 work_dim={}，实际 {}。".format(work_dim, int(bev_feat.shape[1])))
