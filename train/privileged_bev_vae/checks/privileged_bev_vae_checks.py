"""特权 BEV VAE 训练批次校验。

模块: train/privileged_bev_vae/checks/privileged_bev_vae_checks.py
依赖: 无
读取配置: 无
对外接口:
    - check_privileged_bev_vae_batch(batch) -> None
"""


def check_privileged_bev_vae_batch(batch):
    """校验对象: 特权 BEV VAE 训练 batch 的 BEVSeg 通道布局。"""
    if batch["bevseg"].ndim != 4 or tuple(batch["bevseg"].shape[1:]) != (12, 256, 256):
        raise ValueError("特权 BEV VAE batch 期望 [B,12,256,256]")
