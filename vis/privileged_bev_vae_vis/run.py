"""渲染特权 BEV VAE 的 target、重建、latent PCA 和语义细节画布。

模块: vis/privileged_bev_vae_vis/run.py
依赖: argparse, numpy, torch, config, data.bevseg_synthesis, model.privileged_bev_vae,
      vis.bevseg_vis, vis.bevseg_vis.run, vis.privileged_bev_vae_vis.checks
读取配置: privileged_bev_vae_vis.*, data.bevseg
对外接口:
    - main(argv=None) -> None
说明: 场景选择、栅格化和布局复用 BEVSeg 可视化；不修改原 BEVSeg CLI 或输出。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import load_config
from data.bevseg_synthesis import BevSegRasterizer
from model.privileged_bev_vae import PrivilegedBEVVAE
from vis.bevseg_vis import save_privileged_bev_vae_reconstruction
from vis.bevseg_vis.run import (
    _model_input,
    _pick_scene,
    _rasterize_frame,
    _resolve,
    _resolve_device,
)
from vis.data_vis.reader import SceneReader
from vis.privileged_bev_vae_vis.checks.run_checks import check_frame


def _load_checkpoint(model, checkpoint, device):
    """加载 VAE checkpoint；缺失时保留随机权重并明确告警。"""
    path = _resolve(checkpoint)
    if not path.is_file():
        print("[privileged-bev-vae-vis][WARNING] checkpoint 不存在，使用随机权重: {}".format(path))
        return "random"
    payload = torch.load(path, map_location=device, weights_only=True)
    state = payload.get("model", payload)
    model.load_state_dict(state, strict=True)
    epoch = payload.get("epoch", "?") if isinstance(payload, dict) else "?"
    print("[privileged-bev-vae-vis] 已加载权重 {} (epoch={})".format(path, epoch))
    return epoch


def main(argv=None):
    """执行单个真实 BEV 样本的特权 VAE 可视化。"""
    parser = argparse.ArgumentParser(description="特权 BEV VAE 可视化")
    parser.add_argument("--config", default=None)
    parser.add_argument("--env", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--frame", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--prior-samples", type=int, default=None,
                        help="额外渲染的 z~N(0,I) 先验采样数量；0=关闭")
    parser.add_argument("--prior-seed", type=int, default=None,
                        help="先验采样随机种子")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.env)
    vis_cfg = cfg.privileged_bev_vae_vis
    prior_samples = vis_cfg.prior_samples if args.prior_samples is None else args.prior_samples
    prior_seed = vis_cfg.prior_seed if args.prior_seed is None else args.prior_seed
    if prior_samples < 0:
        raise ValueError("--prior-samples 必须 >= 0")
    data_cfg = cfg.data.bevseg
    scene_root = _resolve(data_cfg.scene_root)
    scene_dir = _pick_scene(args.scene if args.scene is not None else vis_cfg.scene, scene_root)
    reader = SceneReader(scene_dir)
    try:
        requested_frame = vis_cfg.frame if args.frame is None else args.frame
        frame = reader.num_frames - 1 if requested_frame == -1 else requested_frame
        check_frame(frame, reader.num_frames)
        rasterizer = BevSegRasterizer(data_cfg)
        sample = _rasterize_frame(reader, rasterizer, frame)
        suffix = "_prior{:02d}".format(prior_samples) if prior_samples else ""
        output = (_resolve(args.output) if args.output else
                  _resolve(vis_cfg.save_dir) /
                  "privileged_bev_vae_{}_f{:06d}{}.png".format(
                      scene_dir.name, frame, suffix))
        device = _resolve_device(vis_cfg.device)
        model = PrivilegedBEVVAE(cfg).to(device).eval()
        epoch = _load_checkpoint(model, args.checkpoint or vis_cfg.checkpoint, device)
        with torch.inference_mode():
            outputs = model(_model_input(sample).to(device), sample=False)
        logits = outputs["reconstruction_logits"][0].float().cpu().numpy()[None]
        latent_mu = outputs["mu"][0].float().cpu().numpy()
        latent_logvar = outputs["logvar"][0].float().cpu().numpy()
        prior_logits = None
        if prior_samples:
            rng_devices = ([device.index if device.index is not None
                            else torch.cuda.current_device()]
                           if device.type == "cuda" else [])
            with torch.random.fork_rng(devices=rng_devices):
                torch.manual_seed(prior_seed)
                prior_z = torch.randn(
                    (prior_samples,) + tuple(outputs["mu"].shape[1:]), device=device)
                with torch.inference_mode():
                    prior_logits = model.decode(prior_z).float().cpu().numpy()[:, None]
        save_privileged_bev_vae_reconstruction(
            [sample], logits, latent_mu, latent_logvar,
            rasterizer.layers, vis_cfg.semantic_threshold, output,
            frame_ids=[frame],
            metadata={"scene": scene_dir.name, "frame": frame, "epoch": epoch,
                      "prior_samples": prior_samples},
            prior_reconstruction_logits=prior_logits,
            prior_sample_ids=list(range(1, prior_samples + 1)) if prior_samples else None)
        print("[privileged-bev-vae-vis] output={}".format(output))
    finally:
        reader.close()


if __name__ == "__main__":
    main()
