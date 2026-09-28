"""特权 BEV VAE 噪声测定输入校验。

模块: tools/privileged_bev_vae_noise/checks/noise_checks.py
依赖: 无
读取配置: 无
对外接口:
    - check_noise_loader(loader) -> None
"""


def check_noise_loader(loader):
    """校验对象: 噪声测定 probe DataLoader。"""
    if len(loader) == 0:
        raise ValueError("噪声测定 probe 数据集不能为空")
