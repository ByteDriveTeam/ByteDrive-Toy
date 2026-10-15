"""整幅 BEV VAE 的输入输出检查复用共享连续 latent VAE 约束。

模块: model/bevseg_vae64/checks/bevseg_vae64_checks.py
依赖: model.privileged_bev_vae.checks.privileged_bev_vae_checks
读取配置: 无
对外接口:
    - check_vae_input(x, in_channels, resolution) -> None
    - check_vae_outputs(outputs, in_channels, latent_dim, latent_resolution, resolution) -> None
"""

from model.privileged_bev_vae.checks.privileged_bev_vae_checks import (
    check_vae_input,
    check_vae_outputs,
)

__all__ = ["check_vae_input", "check_vae_outputs"]
