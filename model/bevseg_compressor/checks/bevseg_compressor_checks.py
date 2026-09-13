"""BEVSeg 压缩器形状与采样检查。"""

import torch


def check_bevseg_input(x, in_channels):
    """检查 BEVSeg 输入是否为五帧堆叠或已展平的四维张量。"""
    if not isinstance(x, torch.Tensor) or x.ndim not in (4, 5):
        raise ValueError("BEVSeg 输入必须为 [B,5,C,H,W] 或 [B,5C,H,W]")
    if x.ndim == 5 and (x.shape[1] != 5 or x.shape[1] * x.shape[2] != in_channels):
        raise ValueError("BEVSeg 历史帧必须为 5 帧且与语义层数匹配")
    if x.ndim == 4 and x.shape[1] != in_channels:
        raise ValueError("BEVSeg 展平输入通道数与配置不一致")
    if x.shape[-2:] != (256, 256):
        raise ValueError("BEVSeg 输入分辨率必须为 256x256")

def check_bevseg_outputs(outputs, in_channels):
    """检查对象: BEVSegCompressor.forward 输出。"""
    if outputs["logits"].shape[2:] != (64, 16):
        raise ValueError("BEVSeg logits 形状错误")
    if outputs["codes"].shape[-1] != 2048:
        raise ValueError("BEVSeg code 必须为 2048 维")
    if outputs["reconstruction_logits"].shape[1] != in_channels:
        raise ValueError("BEVSeg 重建通道数错误")
