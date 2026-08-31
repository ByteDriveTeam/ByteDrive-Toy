import torch


def check_visreg_features(features):
    """校验对象: VISRegLoss 输入 —— 期望有效 batch 至少两样本的 [B,D] 浮点 GAP。"""
    if not isinstance(features, torch.Tensor) or features.ndim != 2:
        raise ValueError("VISReg 特征期望 [B,D]，实际 {}".format(getattr(features, "shape", None)))
    if features.shape[0] < 2 or features.shape[1] < 1 or not torch.is_floating_point(features):
        raise ValueError("VISReg 有效 batch 至少需要两样本，且输入必须为非空浮点特征")
