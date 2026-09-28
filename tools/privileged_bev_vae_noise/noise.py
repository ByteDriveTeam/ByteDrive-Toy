"""测定连续 latent 的噪声、冗余维度和扰动敏感度。

模块: tools/privileged_bev_vae_noise/noise.py
依赖: csv, json, pathlib, torch, torch.utils.data
读取配置: train.privileged_bev_vae.noise.*
对外接口:
    - measure_latent_noise(model, dataset, cfg, device, epoch=0) -> dict
说明: 使用固定的顺序 probe 子集；统计 posterior、协方差谱和逐维噪声扰动，
      结果写入 JSON/CSV，不自动修改 latent 维度。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from tools.privileged_bev_vae_noise.checks.noise_checks import check_noise_loader


def _repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else Path(__file__).resolve().parents[2] / path


def _write_reports(summary, output_dir, epoch):
    output_dir = _repo_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "epoch_{:03d}".format(epoch + 1)
    (output_dir / (stem + ".json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / (stem + ".csv")).open("w", newline="", encoding="utf-8") as handle:
        fields = ["dimension", "mean", "std", "posterior_std", "kl", "perturbation_abs",
                  "perturbation_relative", "near_zero"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary["per_dimension"])


@torch.no_grad()
def measure_latent_noise(model, dataset, cfg, device, epoch=0):
    """在固定 probe 子集上测量 posterior 噪声和 latent 维度冗余。"""
    noise_cfg = cfg.train.privileged_bev_vae.noise
    limit = min(int(noise_cfg.max_samples), len(dataset))
    probe = DataLoader(
        Subset(dataset, range(limit)), batch_size=noise_cfg.batch_size,
        shuffle=False, num_workers=0, drop_last=False)
    check_noise_loader(probe)
    was_training = model.training
    model.eval()
    means, stds, posterior_stds, kls, sensitivity, relative = [], [], [], [], [], []
    covariance_samples = []
    all_sensitivity, all_relative = [], []
    rng_devices = ([device.index if device.index is not None else torch.cuda.current_device()]
                   if device.type == "cuda" else [])
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(int(noise_cfg.seed) + int(epoch))
        for batch in probe:
            x = batch["bevseg"].to(device, non_blocking=True)
            encoded = model.encode(x, sample=False)
            mu = encoded["mu"]
            logvar = encoded["logvar"]
            posterior_std = torch.exp(0.5 * logvar)
            flat_mu = mu.permute(0, 2, 3, 1).reshape(-1, mu.shape[1])
            covariance_samples.append(flat_mu.float().cpu())
            means.append(flat_mu.mean(0).cpu())
            stds.append(flat_mu.std(0, unbiased=False).cpu())
            posterior_stds.append(posterior_std.permute(0, 2, 3, 1).reshape(-1, mu.shape[1]).mean(0).cpu())
            kls.append((0.5 * (mu.square() + logvar.exp() - 1.0 - logvar)).mean((0, 2, 3)).cpu())
            baseline = model.decode(mu)
            all_perturb = torch.randn_like(mu) * posterior_std * noise_cfg.noise_scale
            all_changed = model.decode(mu + all_perturb)
            all_delta = (all_changed - baseline).float()
            all_sensitivity.append(all_delta.abs().mean().cpu())
            all_relative.append((all_delta.square().mean().sqrt() /
                                 baseline.float().square().mean().sqrt().clamp_min(1e-8)).cpu())
            channel_abs = []
            channel_rel = []
            for dim in range(mu.shape[1]):
                perturb = torch.zeros_like(mu)
                perturb[:, dim] = torch.randn_like(mu[:, dim]) * posterior_std[:, dim] * noise_cfg.noise_scale
                changed = model.decode(mu + perturb)
                delta = (changed - baseline).float()
                channel_abs.append(delta.abs().mean().cpu())
                channel_rel.append((delta.square().mean().sqrt() /
                                    baseline.float().square().mean().sqrt().clamp_min(1e-8)).cpu())
            sensitivity.append(torch.stack(channel_abs))
            relative.append(torch.stack(channel_rel))
    if was_training:
        model.train()
    mu_samples = torch.cat(covariance_samples, dim=0)
    centered = mu_samples - mu_samples.mean(0, keepdim=True)
    covariance = centered.T @ centered / max(centered.shape[0] - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0).flip(0)
    eigen_sum = eigenvalues.sum().clamp_min(1e-12)
    participation = eigen_sum.square() / eigenvalues.square().sum().clamp_min(1e-12)
    mean = torch.stack(means).mean(0)
    std = torch.stack(stds).mean(0)
    posterior_std = torch.stack(posterior_stds).mean(0)
    kl = torch.stack(kls).mean(0)
    abs_delta = torch.stack(sensitivity).mean(0)
    relative_delta = torch.stack(relative).mean(0)
    per_dimension = [{
        "dimension": index,
        "mean": float(mean[index]),
        "std": float(std[index]),
        "posterior_std": float(posterior_std[index]),
        "kl": float(kl[index]),
        "perturbation_abs": float(abs_delta[index]),
        "perturbation_relative": float(relative_delta[index]),
        "near_zero": bool(std[index] <= noise_cfg.near_zero_std_threshold),
    } for index in range(mean.numel())]
    summary = {
        "epoch": int(epoch + 1),
        "samples": int(limit),
        "latent_dim": int(mean.numel()),
        "effective_rank": float(participation),
        "eigenvalues": [float(value) for value in eigenvalues],
        "near_zero_count": sum(row["near_zero"] for row in per_dimension),
        "noise_scale": float(noise_cfg.noise_scale),
        "all_latent_perturbation_abs": float(torch.stack(all_sensitivity).mean()),
        "all_latent_perturbation_relative": float(torch.stack(all_relative).mean()),
        "per_dimension": per_dimension,
    }
    _write_reports(summary, noise_cfg.output_dir, epoch)
    return summary
