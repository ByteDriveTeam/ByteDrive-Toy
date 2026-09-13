# BEVSeg compressor contract

The BEVSeg task consumes only driving-relevant labels produced from CARLA collector metadata and the HD map. It does not consume LiDAR.

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

The encoder uses a 16×16 stride-16 projection to 384 channels, residual blocks at 16×16, 8×8, and 4×4, and grouped logits with 64 independent 16-entry subword vocabularies of dimension 32. Gumbel categorical sampling uses a straight-through estimator. During the first 20% of training, the active candidate set anneals from Top-4 to Top-1 while temperature decreases.

The decoder expands 4×4 codes to 256×256 through six 2× PixelShuffle stages. Expand convolutions use ICNR initialization; spatial convolutions and residual channel mixing retain capacity while avoiding interpolation blur and reducing initial checkerboard artifacts.
