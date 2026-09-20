"""BEVSeg 压缩器形状与采样检查。"""

import torch


def check_bevseg_input(x, in_channels):
    """检查对象: BEVSegCompressor.encode 输入 —— 单帧四维张量。"""
    if not isinstance(x, torch.Tensor) or x.ndim != 4:
        raise ValueError("BEVSeg 输入必须为 [B,C,H,W] 单帧张量")
    if x.shape[1] != in_channels:
        raise ValueError("BEVSeg 输入通道数与配置不一致")
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
