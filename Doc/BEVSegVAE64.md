# BEVSeg 64x continuous latent VAE

`bevseg_vae64` is an additional VAE task. The existing `privileged_bev_vae`
configuration and checkpoints remain unchanged.

The model keeps the BEVSeg contract `[B,12,256,256]` and uses six stride-2
encoder stages, producing a `[B,64,4,4]` Gaussian latent (`mu`, `logvar`, and
`z`). Six ICNR initialized PixelShuffle stages restore the 256x256 logits.

Run the additional task with:

```powershell
.\.venv\Scripts\python.exe -m train.run --task bevseg_vae64
```

Its checkpoints are written below `train/ckpt/bevseg_vae64/`. Reconstruction
losses and the KL regularizer reuse the existing BEV VAE training path.
