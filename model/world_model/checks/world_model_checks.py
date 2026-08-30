import torch


def check_grid_input(grids, channels, frames, height, width):
    """校验对象: WorldModel 输入 grids —— 必须为 [B,T,C,H,W] 且通道/时空尺寸匹配配置。"""
    expected = (frames, channels, height, width)
    if not isinstance(grids, torch.Tensor) or grids.ndim != 5 or tuple(grids.shape[1:]) != expected:
        raise ValueError("grids 期望 [B,{}]，实际 {}".format(
            ",".join(str(v) for v in expected), getattr(grids, "shape", None)))


def check_spatial_mask(mask, batch_size, patch_count):
    """校验对象: Student 空间 mask —— bool [B,P] 且每个样本可见 Token 数一致且非零。"""
    if mask.dtype != torch.bool or tuple(mask.shape) != (batch_size, patch_count):
        raise ValueError("mask 期望 bool [{},{}]，实际 {} {}".format(
            batch_size, patch_count, tuple(mask.shape), mask.dtype))
    visible = (~mask).sum(1)
    if int(visible.min()) <= 0 or not bool(torch.equal(visible, visible[:1].expand_as(visible))):
        raise ValueError("每个样本必须有相同且非零的可见空间 Token 数")
