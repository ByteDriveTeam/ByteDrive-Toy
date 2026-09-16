# BEVSeg compressor contract

The BEVSeg task consumes only driving-relevant labels produced from CARLA collector metadata and the HD map. It does not consume LiDAR.

## Dataset windows

Each sample contains five consecutive stored frames ordered from oldest to current, all rasterized in the current frame's ego coordinate system. Window endpoints are sampled once per `data.bevseg.window_stride_s` using each scene's recorded `sensor_dt_s`; the default one-second stride is therefore ten stored frames for 10 Hz scenes. Windows never cross scene boundaries, and only scenes with fewer than five stored frames are excluded. A scene-level `failed` flag does not exclude otherwise complete windows because compressor training benefits from the additional state diversity.

Because all five frames share that current-frame coordinate system, the HD-map drivable mask, lane classes, lane directions, and static boxes are projected once per window and copied before frame-varying dynamic boxes and traffic controls are drawn. The batched path is pixel-equivalent to independently rasterizing the five frames.

## Raster cache

The dataset uses a bounded read-through cache at `data.bevseg.cache.dir`. A cache hit reconstructs the same `float32` tensors without loading frame metadata or running HD-map/box rasterization. Binary semantic layers are bit-packed, while the lane-direction field remains `float32` and is stored once because all five frames share it. Entries use compressed NumPy `.npz` files and atomic writes, so DataLoader workers may safely fill missing entries while training. Each worker retains every HD map it has loaded, keyed by map path rather than scene, so scenes from the same town never reload and reparse that map.

`data.bevseg.cache.max_size_gb` caps the payload across cache versions and defaults to 32 GiB. Once the next entry would exceed the limit, the cache keeps all existing entries and stops adding new ones; misses are still rasterized normally. `compression_level` controls the zlib speed/size tradeoff. Raster configuration, rasterizer source, source LMDB size/time, and HD-map size/time participate in cache identity so changed inputs do not silently reuse stale labels.

With the default `prebuild: true`, normal BEVSeg training first scans the cache, reports the current sample number and percentage every `progress_every` samples, and generates only missing entries before constructing the model. The cache scan and generation cover the same complete `BevSegDataset` consumed by training. The project has no separate train/validation/test split; batching-time shuffle and `drop_last` do not define a stable subset, so prebuild intentionally covers every sample that may appear in any epoch.

To populate the same cache before training without constructing the model:

```powershell
.\.venv\Scripts\python.exe -m train.run --task bevseg --prepare-bevseg-cache
```

The command uses `train.num_workers`, `train.prefetch_factor`, and `train.in_order`, reports progress every `data.bevseg.cache.progress_every` samples, and stops when the configured capacity is full. Set `prebuild: false` to skip startup pre-generation while retaining training-time read-through writes. Set `write_missing: false` together with `prebuild: false` for read-only use, or `enabled: false` to bypass the cache.

During training, BEVSeg prints the current epoch, step, percentage, and every loss component at step 1, every `train.log_every` steps, and the final step. The epoch-end line remains a sample-weighted aggregate. Loss accumulation stays on the training device between log points to avoid a device synchronization for every component on every batch.

## Coordinate and raster contract

- Public BEV axes: `X` is right-positive, `Y` is front-positive, `Z` is up.
- The ego vehicle is at the intersection of the two image diagonals (`X=0, Y=0` in the metric frame), i.e. pixel center `(128, 128)`.
- The square covers 64 m × 64 m (`[-32, 32]` m on each public axis) at 256 × 256 pixels.
- Row increases toward the rear and column increases toward the right:

  `row = (1 - (Y + 32) / 64) * 256`, `col = (X + 32) / 64 * 256`.

The existing numeric CARLA ego transform uses `(forward, right)` for its planar pair. The rasterizer passes that pair to the shared `ego_xy_to_pixel` utility; public `(X, Y)` is therefore `(right, forward)` without changing the established transform implementation.

## Semantic channels

Each frame has ten binary channels, in this fixed order:

`drivable`, `lane_centerline`, `lane_divider`, `road_boundary`, `pedestrian_crossing`, `vehicle`, `pedestrian`, `stop_line_red`, `stop_line_yellow`, `stop_line_green`.

The final three channels are a fused stop-line/control representation. No standalone pole, traffic-light, or untyped stop-line channel is emitted. Two additional continuous channels store lane tangent direction `(right, front)` and are masked by lane pixels. Five ego-aligned frames are concatenated for a 60-channel model input.

## Compression path

Training also applies entropy regularization to the processed categorical distribution, a per-subword concentration loss to each complete 16-way probability distribution before Top-8 sampling, and a usage-balance loss that maximizes the batch-and-spatial marginal entropy of each 16-entry subword vocabulary.

The encoder uses a 16×16 stride-16 projection to 384 channels, residual blocks at 16×16, 8×8, and 4×4, and grouped logits with 64 independent 16-entry subword vocabularies of dimension 32. During training, each subword vocabulary first takes its probability-ranked Top-8 candidates, keeps the global Top-1, and samples three additional candidates without replacement from the Top-8 probabilities. The sampled Top-4 probabilities are linearly normalized and linearly sharpened toward a straight-through Top-1 one-hot selection over the first 20% of training. The resulting per-vocabulary probabilities are used to form a weighted codebook vector; inference always uses deterministic Top-1 hard selection. After the 64 subword vectors are concatenated, RMSNorm is applied over the complete 2048-dimensional code before it is returned by the encoder or consumed by the decoder.

The decoder expands 4×4 codes to 256×256 through six 2× PixelShuffle stages. Expand convolutions use ICNR initialization; spatial convolutions and residual channel mixing retain capacity while avoiding interpolation blur and reducing initial checkerboard artifacts.
