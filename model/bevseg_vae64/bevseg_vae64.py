"""整幅 BEV 的连续 latent VAE：将 256×256 栅格压缩到 4×4。

模块: model/bevseg_vae64/bevseg_vae64.py
依赖: torch, model.privileged_bev_vae
读取配置: model.bevseg_vae64.*, data.bevseg.resolution
对外接口:
    - BEVSegVAE64(cfg) -> nn.Module
      encode(x, sample=None) -> dict
      decode(z) -> Tensor
      forward(x, sample=None) -> dict
说明: 复用现有 BEV VAE 的连续高斯 latent、残差块和 PixelShuffle，增加两级
      stride-2 下采样与上采样，使空间尺寸从 256×256 变为 4×4。
"""

from model.privileged_bev_vae import PrivilegedBEVVAE

__all__ = ["BEVSegVAE64"]


class BEVSegVAE64(PrivilegedBEVVAE):
    """将整幅 BEVSeg 栅格压缩为 64 通道、4×4 空间 latent。"""

    def __init__(self, cfg) -> None:
        super().__init__(cfg, config_key="bevseg_vae64")
