import torch


def check_visreg_features(features):
    """校验对象: VISRegLoss 输入 —— 期望至少两视图、两样本的 [V,B,D] 浮点 GAP。"""
    if not isinstance(features, torch.Tensor) or features.ndim != 3:
        raise ValueError("VISReg 特征期望 [V,B,D]，实际 {}".format(getattr(features, "shape", None)))
    if features.shape[0] < 2 or features.shape[1] < 2 or not torch.is_floating_point(features):
        raise ValueError("VISReg 至少需要两视图、两样本且输入必须为浮点")
