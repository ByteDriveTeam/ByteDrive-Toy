"""特权 BEV VAE latent 噪声测定命令行入口。

模块: tools/privileged_bev_vae_noise/run.py
依赖: argparse, torch, config, data.bevseg_dataset, model.privileged_bev_vae, tools.privileged_bev_vae_noise
读取配置: train.device, train.privileged_bev_vae.noise.*
对外接口:
    - main(argv=None) -> None
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import load_config
from data.bevseg_dataset import BevSegDataset
from model.privileged_bev_vae import PrivilegedBEVVAE
from tools.privileged_bev_vae_noise.noise import measure_latent_noise


def main(argv=None):
    """加载检查点并执行一次 latent 噪声测定。"""
    parser = argparse.ArgumentParser(description="特权 BEV VAE latent 噪声测定")
    parser.add_argument("--config", default=None)
    parser.add_argument("--env", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--epoch", type=int, default=0)
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.env)
    device = torch.device(cfg.train.device if torch.cuda.is_available() or
                          not cfg.train.device.startswith("cuda") else "cpu")
    model = PrivilegedBEVVAE(cfg).to(device)
    checkpoint = torch.load(Path(args.checkpoint), map_location=device)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
    dataset = BevSegDataset(cfg)
    summary = measure_latent_noise(model, dataset, cfg, device, args.epoch)
    print("[privileged_bev_vae-noise] epoch={} effective_rank={:.3f} near_zero={}".format(
        summary["epoch"], summary["effective_rank"], summary["near_zero_count"]))


if __name__ == "__main__":
    main()
