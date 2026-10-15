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

## Visualization

The VAE keeps the existing BEVSeg target/reconstruction/detail layout while
replacing the discrete-code PCA panel with PCA of the spatial `latent_mu`
(`16×16` by default). The summary panel also shows effective rank, posterior
standard deviation, and raw KL.

```powershell
.\.venv\Scripts\python.exe -m vis.privileged_bev_vae_vis.run `
  --checkpoint train/ckpt/privileged_bev_vae/epoch_010.pt `
  --scene scene_000000 --frame 100
```

To inspect whether the standard-normal prior decodes to plausible BEV layouts,
append a small number of prior samples. The output keeps the same target and
reconstruction rows, then adds one composite/detail group per sampled latent:

```powershell
.\.venv\Scripts\python.exe -m vis.privileged_bev_vae_vis.run `
  --checkpoint train/ckpt/privileged_bev_vae/epoch_033.pt `
  --scene scene_000000 --frame 100 --prior-samples 4 --prior-seed 20260928
```

The default `prior_samples: 0` leaves the original VAE visualization unchanged.
Each prior sample uses `z ~ N(0,I)` and is decoded without posterior statistics;
its semantic mIoU, direction cosine, and semantic MAE are shown against the
selected target frame.

The original `vis.bevseg_vis` command and its output layout remain unchanged.
