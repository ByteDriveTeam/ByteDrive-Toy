# 本文件为 model/bev_encoder/bev_encoder.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_bev_encoder_inputs(bev_query, image_feat, work_dim, previous_bev=None,
                             previous_geometry=None, previous_valid=None):
    """校验对象: BevEncoder.forward 入参 —— 当前查询/图像与可选历史 BEV 的形状、通道、批次须一致。"""
    if bev_query.ndim != 4 or image_feat.ndim != 4:
        raise ValueError("bev_query / image_feat 期望 (B,C,H,W) 四维，实际 {} / {}。".format(
            tuple(bev_query.shape), tuple(image_feat.shape)))
    if int(bev_query.shape[1]) != work_dim or int(image_feat.shape[1]) != work_dim:
        raise ValueError("bev_query / image_feat 通道须等于 work_dim={}，实际 {} / {}。".format(
            work_dim, int(bev_query.shape[1]), int(image_feat.shape[1])))
    if int(bev_query.shape[0]) != int(image_feat.shape[0]):
        raise ValueError("bev_query 与 image_feat 批次须一致，实际 {} / {}。".format(
            int(bev_query.shape[0]), int(image_feat.shape[0])))
    history = (previous_bev, previous_geometry, previous_valid)
    if all(item is None for item in history):
        return
    if any(item is None for item in history):
        raise ValueError("previous_bev / previous_geometry / previous_valid 必须同时提供或同时省略。")
    if previous_bev.shape != bev_query.shape or previous_geometry.shape != bev_query.shape:
        raise ValueError("历史 BEV/几何须与 bev_query 同形，实际 {} / {} / {}。".format(
            tuple(previous_bev.shape), tuple(previous_geometry.shape), tuple(bev_query.shape)))
    if previous_valid.ndim != 1 or int(previous_valid.shape[0]) != int(bev_query.shape[0]):
        raise ValueError("previous_valid 期望 [B]，实际 {}。".format(tuple(previous_valid.shape)))
