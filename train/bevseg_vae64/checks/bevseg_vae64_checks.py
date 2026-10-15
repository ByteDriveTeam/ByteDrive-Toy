"""整幅 BEV VAE 训练批次检查复用现有 BEV VAE 约束。

模块: train/bevseg_vae64/checks/bevseg_vae64_checks.py
依赖: train.privileged_bev_vae.checks.privileged_bev_vae_checks
读取配置: 无
对外接口:
    - check_privileged_bev_vae_batch(batch) -> None
"""

from train.privileged_bev_vae.checks.privileged_bev_vae_checks import (
    check_privileged_bev_vae_batch,
)

__all__ = ["check_privileged_bev_vae_batch"]
