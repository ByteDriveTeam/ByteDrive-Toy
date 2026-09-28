# Privileged BEV continuous-latent VAE

`privileged_bev_vae` reuses the existing BEVSeg raster and cache, but replaces
the discrete codebook with a configurable continuous Gaussian latent. The input
remains `[B,12,256,256]` in the ego-aligned `64m × 64m` BEV (`X=right`,
`Y=front`).

## Architecture

The encoder uses four stride-2 `2×2` convolutions with residual block counts
`[2,3,2,1]`, producing a `16×16` spatial bottleneck. RMSNorm2d is applied
before the mean and log-variance projections. The default latent is
`[B,24,16,16]`.

The decoder consumes the latent directly, with no normalization before its
first upsampling, and uses four ICNR-initialized PixelShuffle stages with
`[256,192,128,96]` channels before producing 12 reconstruction logits.

## Loss and training

Training uses the BEVSeg weighted semantic BCE and lane-direction SmoothL1
supervision, plus the configurable Gaussian KL term:

`total = semantic + direction_weight × direction + kl_weight × KL`.

Run the task with:

```powershell
.\.venv\Scripts\python.exe -m train.run --task privileged_bev_vae
```

The existing BEVSeg raster cache is prebuilt and reused when enabled. Checkpoints
are written below `train/ckpt/privileged_bev_vae/`.

## Latent noise measurement

When `train.privileged_bev_vae.noise.enabled` is true, the trainer runs a fixed
sequential probe after every epoch. It writes one JSON summary and one per-dimension
CSV below the configured output directory. Reports contain posterior statistics,
per-dimension KL, covariance eigenvalues, effective rank, near-zero variance
dimensions, and reconstruction sensitivity to independent latent perturbations.

The same measurement is available independently:

```powershell
.\.venv\Scripts\python.exe -m tools.privileged_bev_vae_noise.run `
  --checkpoint train/ckpt/privileged_bev_vae/epoch_010.pt
```

The tool reports redundancy only; it never changes `latent_dim` or model weights.
