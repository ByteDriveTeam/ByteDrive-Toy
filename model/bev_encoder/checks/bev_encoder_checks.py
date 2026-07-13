# 本文件为 model/bev_encoder/bev_encoder.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_bev_encoder_inputs(bev_query, image_feat, work_dim):
    """校验对象: BevEncoder.forward 入参 —— 查询/图像特征均 [B,work_dim,H,W] 四维、通道等于 work_dim、批次一致。"""
    if bev_query.ndim != 4 or image_feat.ndim != 4:
        raise ValueError("bev_query / image_feat 期望 (B,C,H,W) 四维，实际 {} / {}。".format(
            tuple(bev_query.shape), tuple(image_feat.shape)))
    if int(bev_query.shape[1]) != work_dim or int(image_feat.shape[1]) != work_dim:
        raise ValueError("bev_query / image_feat 通道须等于 work_dim={}，实际 {} / {}。".format(
            work_dim, int(bev_query.shape[1]), int(image_feat.shape[1])))
    if int(bev_query.shape[0]) != int(image_feat.shape[0]):
        raise ValueError("bev_query 与 image_feat 批次须一致，实际 {} / {}。".format(
            int(bev_query.shape[0]), int(image_feat.shape[0])))
