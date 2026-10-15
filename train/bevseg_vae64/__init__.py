"""整幅 BEV 64 倍压缩 VAE 的训练公开 API。"""

from train.bevseg_vae64.bevseg_vae64 import train_bevseg_vae64_epoch

__all__ = ["train_bevseg_vae64_epoch"]
