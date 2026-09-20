"""BEVSeg 训练输入检查。"""

def check_bevseg_batch(batch):
    """检查对象: BEVSeg batch 的单帧和通道布局。"""
    if batch["bevseg"].ndim != 4 or batch["bevseg"].shape[1:] != (12, 256, 256):
        raise ValueError("BEVSeg batch 期望 [B,12,256,256]")
