# ByteDrive-Toy

<p align="center">
  <img alt="ByteDrive logo" src="assets/logo_full.png" width="391">
</p>

<p align="center">
  <strong>A research prototype for multi-task vision-centric autonomous driving, privileged BEV representations, and behavior-cloning on synthetic CARLA data</strong><br/>
  Frozen DINOv3 backbone · semantic/depth pretraining · BEVSeg grouped discrete codebook & continuous VAE compression · unified 4-frame flow-matching driving · CARLA closed-loop evaluation
</p>


<p align="center">
  <a href="README.zh-CN.md">简体中文</a> · <strong>English</strong>
</p>

<p align="center">
  <img alt="Python 3.12" src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white">
  <img alt="CARLA 0.9.15" src="https://img.shields.io/badge/CARLA-0.9.15-00A6D6">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-BF16%20%2B%20FP32-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="Apache-2.0 license" src="https://img.shields.io/badge/License-Apache--2.0-blue.svg">
</p>

> [!IMPORTANT]
> ByteDrive-Toy covers CARLA data collection, perception pretraining, BEV driving training, offline prediction and visualization, and **behavior-cloning closed-loop driving in CARLA 0.9.15**. `clone_loop/` performs two-frame inference, safety-aware trajectory reranking, vehicle control, episode evaluation, and recording in synchronous simulation. This remains a simulation research prototype, not a complete autonomous-driving system for deployment on real vehicles.

> [!WARNING]
> The driving model is trained by supervised **behavior cloning** on CARLA `BehaviorAgent` expert trajectories. This repository does not implement reward-based reinforcement learning, online policy updates, replay, or backpropagation from closed-loop episodes. The current data and released weights are concentrated on `Town02_Opt`, one expert style, and a fixed collection recipe. Distribution shift and compounding error are expected; stopping, drifting, collisions, low route completion, and weak cross-map generalization do not by themselves indicate a broken installation.

## Contents

- [Overview](#overview)
- [Visual results](#visual-results)
- [System architecture](#system-architecture)
- [From pixels to steering](#from-pixels-to-steering)
- [Model architecture](#model-architecture)
- [Data system](#data-system)
- [Training](#training)
- [Pretrained weights](#pretrained-weights)
- [Installation](#installation)
- [Quick start](#quick-start)
- [End-to-end workflow](#end-to-end-workflow)
- [Configuration](#configuration)
- [Visualization](#visualization)
- [Repository layout](#repository-layout)
- [Limitations and troubleshooting](#limitations-and-troubleshooting)
- [Development and license](#development-and-license)

## Overview

ByteDrive-Toy organizes the vision-centric driving research pipeline into multiple progressive learning and representation stages:

1. **Perception pretraining.** A frozen local DINOv3 ViT-S+/16 backbone supplies full token sequences across shallow, middle, and deep layers. A shared trainable trunk jointly learns 29-class semantic segmentation and metric depth estimation.
2. **Privileged BEV representation and compression.**
   - **BEVSeg compressor.** Renders 64 m × 64 m (256 × 256) HDMap, dynamic bounding boxes, and traffic-control states into driving-relevant semantic rasters. A 384-dimensional residual encoder, 64-table grouped discrete codebook, and ICNR PixelShuffle decoder achieve 16x spatial compression with bounded caching ([Doc/BEVSeg.md](Doc/BEVSeg.md)).
   - **Continuous latent VAEs.** A 4-stage continuous Gaussian latent privileged BEV VAE ([Doc/PrivilegedBEVVAE.md](Doc/PrivilegedBEVVAE.md)) and a full-BEV 64x continuous-latent VAE compressing 256 × 256 rasters into a compact 4 × 4 latent ([Doc/BEVSegVAE64.md](Doc/BEVSegVAE64.md)), with latent noise, covariance, and effective-rank measurement tools.
3. **Driving learning.**
   - **Unified driving model (`DrivingModel` / `UnifiedDrivingModel`).** Four-frame monocular input, 12-layer block-causal visual processing, 512 local planar acceleration words, standardized residual endpoint matching, and a separate continuous BEV flow decoder ([model contract](Doc/UnifiedDrivingModel.md)).
4. **Trajectory vocabulary and 3D reconstruction.**
   - Streaming CARLA ego kinematics extraction, 0.5 s resampled feature computation, and K-Means clustering into geometric and velocity vocabularies with static visualization.
   - Multi-frame semantic LiDAR fusion, static/dynamic object separation, and sparse TUDF / optional watertight Poisson mesh generation.
5. **Closed-loop simulation evaluation.**
   - The existing synchronous CARLA 0.9.15 runner (`clone_loop/`) retains its earlier two-frame model input contract; it needs an input/output adapter and a compatible checkpoint before it can evaluate the unified model.

| Capability | Current implementation |
| --- | --- |
| Learning paradigm | Supervised behavior cloning and flow matching on CARLA `BehaviorAgent` data; no online RL policy updates |
| Visual & temporal input | 4-frame forward monocular RGB sequences (`UnifiedDrivingModel`); or 2-frame 3-camera `front/front_left/front_right` RGB (baseline) |
| Visual backbone | Local frozen DINOv3 ViT-S+/16; full token sequences from layers 3, 6, and 12 |
| Perception tasks | 29-class semantic segmentation, symlog depth regression, and in-range/out-of-range depth classification |
| Driving modeling | 4-frame spatio-temporal token trunk + AdaLN-Zero dual flow-matching (Signed-Log space); or 2-frame BEV Transformer + 8 learned planning tokens |
| Privileged BEV | 64 m × 64 m (256 × 256) driving semantic raster; 64-table grouped discrete codebook compressor; 4-stage continuous Gaussian latent VAE; 64x spatial compression VAE |
| Geometry & space | Camera intrinsics/extrinsics back-projection; patch-depth frustums; zero-continuous Signed-Log physical metric space |
| Trajectory vocabulary | Streaming extraction and K-Means clustering of CARLA ego geometric trajectory and velocity profiles |
| Training CLI | Unified `python -m train.run --task <task>` for `perception`, `driving`, `bevseg` (with cache warmup), `privileged_bev_vae`, `bevseg_vae64` |
| Reconstruction | Multi-frame semantic LiDAR fusion; static/dynamic separation; sparse TUDF by default and optional Poisson watertight mesh |
| Closed loop | Synchronous CARLA 0.9.15, online inference, safety reranking, pure-pursuit lateral control, speed PID, episode metrics, and dual video recording |

```mermaid
flowchart LR
    A["CARLA collection"] --> B["Data inspection"]
    B --> C["Perception pretraining"]
    B --> V1["BEVSeg raster synthesis"]
    V1 --> V2["Grouped codebook & VAE compression"]
    B --> TV["Trajectory & velocity vocabulary"]
    B --> R1["Semantic LiDAR fusion"]
    R1 --> R2["Sparse TUDF / optional mesh"]
    C --> D["Unified flow-matching & BEV driving"]
    V2 -.->|"privileged latents"| D
    D --> E["Offline visualization & noise diagnostics"]
    D --> F["Behavior-cloning closed loop"]
    F --> G["CARLA control and evaluation"]
    G -.->|"not implemented"| H["Closed-loop reinforcement learning"]
```

## Visual results

### CARLA closed-loop driving

The videos show online `clone_loop` episodes. The top row contains the left/front/right cameras and vehicle state; the middle row shows risk, drivable, trajectory-distribution, lane, and traffic-control predictions; the bottom row shows all candidates and the executed trajectory.

<video src="assets/visualizations/driving_video_1.mp4" controls="controls" width="768"></video>
<video src="assets/visualizations/driving_video_2.mp4" controls="controls" width="768"></video>

[Closed-loop example 1](assets/visualizations/driving_video_1.mp4) · [Closed-loop example 2](assets/visualizations/driving_video_2.mp4)

### Perception predictions

Each column is one frame. From top to bottom: RGB, predicted semantics, predicted depth, semantic ground truth, and depth ground truth.

![ByteDrive perception predictions](assets/visualizations/perception.png)

### Driving predictions

The renderer compares RGB/semantic/depth views, the three BEV fields, lane geometry, traffic controls, and multimodal trajectories against ground truth with identical geometry and colors.

![ByteDrive driving predictions](assets/visualizations/driving.png)

The traffic-control view colors stop lines by red/yellow/green state and overlays them on the trajectory BEV.

![ByteDrive stop-line and traffic-light ground truth](assets/visualizations/traffic_control_gt.png)

## System architecture

```mermaid
flowchart TB
    subgraph SIM["CARLA synthetic data"]
        C1["BehaviorAgent ego vehicle"]
        C2["TrafficManager vehicles and pedestrians"]
        C3["RGB · depth · semantics · optical flow · LiDAR"]
        C1 --> C3
        C2 --> C3
    end
    subgraph STORE["Heterogeneous collection and storage"]
        W["Python 3.7 worker<br/>CARLA 0.9.15"]
        SHM["Shared-memory arena<br/>zero-copy frame handoff"]
        COL["Python 3.12 collector<br/>orchestration, encoding, storage"]
        MP4["RGB -> H.265 MP4"]
        DB["Other modalities and labels -> LMDB"]
        W --> SHM --> COL
        COL --> MP4
        COL --> DB
    end
    subgraph LEARN["Learning"]
        P["PerceptionModel"]
        D["DrivingModel"]
        P -->|"initialize perception representation"| D
    end
    subgraph CLOSED["Behavior-cloning closed loop"]
        OBS["Three RGB views + LiDAR<br/>pose, velocity, route target"]
        INF["Two-frame inference"]
        SELECT["Confidence + risk + drivable<br/>+ route-alignment reranking"]
        CTRL["Pure pursuit + speed PID"]
        STEP["apply_control -> world.tick"]
        OBS --> INF --> SELECT --> CTRL --> STEP --> OBS
    end
    SIM --> W
    MP4 --> P
    DB --> P
    MP4 --> D
    DB --> D
    D --> INF
```

Configuration, implementation, and runtime validation are separated: experiment parameters live in `config/default.yaml`, module implementations live in their package directories, and input/shape checks live in adjacent `checks/` directories.

## From pixels to steering

The central idea is to preserve both appearance and geometry. The visual backbone explains *what* is in each patch; camera calibration and depth candidates explain *where it may be*; the BEV encoder turns those observations into a local world representation; the planning decoder turns that representation into several plausible futures.

### Perception first

Semantic and depth pretraining gives the visual trunk dense, physically meaningful supervision before driving optimization. The frozen DINOv3 backbone supplies general visual features, while trainable fusion and transformer layers adapt them to CARLA semantics and metric depth. Driving training reuses only the shared `fusion.*` and `trunk.*` parameters—not the semantic/depth output heads.

### Camera and LiDAR geometry

For each camera patch, the model encodes possible 3D locations by combining the patch center and four corners with near-to-far depth samples. The three cameras keep their own calibration. BEV queries start from metric xyz grid centers and cross-attend to image tokens. LiDAR is voxelized in ego coordinates and represented by the mean and population standard deviation of center-relative xyz values. A visual-conditioned gate injects those statistics before camera cross-attention.

### Temporal alignment

The previous BEV cannot be reused in its original coordinate frame after the vehicle moves. `previous_to_current` rigidly maps historical BEV coordinates into the current ego frame. The current BEV queries then attend to aligned historical tokens; `previous_valid=0` strictly bypasses missing history.

### Joint spatial and planning objectives

Dense fields and lane/traffic predictions force the BEV to represent road structure, visibility, and hazards. Eight learned mode tokens query intermediate BEV features while conditioned on the navigation target and ego velocity. The decoder predicts trajectories, confidence, and behavior labels. At runtime, neural confidence is combined with sampled risk, drivable probability, and route alignment before a low-level controller executes the winner.

Behavior cloning alone does not train recovery from arbitrary states. Offline losses optimize agreement with expert data, while closed-loop success depends on stability under the model's own state distribution. Broader data, recovery perturbations, DAgger-style aggregation, or a separately implemented closed-loop optimization method are needed to address that gap.

## Model architecture

### Tensor overview

| Stage | Typical tensor or resolution |
| --- | --- |
| Camera input | 3 views x RGB, `768x384`, current and previous frames |
| DINO patch grid | `48x24` per view at patch size 16 |
| Perception outputs | semantic/depth at input resolution |
| LiDAR voxel grid | statistics `[6,22,128,128]`, occupancy `[1,22,128,128]` |
| BEV working grid | `32x32` over the forward `64 m x 64 m` region |
| Dense BEV outputs | `256x256` |
| Trajectories | 8 modes x 20 ego-frame waypoints covering 2 seconds |

### Perception model

1. The local DINOv3 ViT-S+/16 is loaded with `local_files_only=True`, held in `eval()` mode, and permanently frozen.
2. Full token sequences from configured layers are RMS-normalized, concatenated, and projected to the working dimension.
3. Three pre-norm transformer layers with patch-only 2D RoPE adapt the representation.
4. PixelShuffle heads decode 29 semantic classes and two depth channels: symlog metric depth plus an in-range logit.

### Driving model

1. **Driving neck:** combines the adapted perception tokens with raw DINO features and frustum geometry.
2. **BEV initialization:** metric xyz grid centers define geometry-only queries; gated LiDAR features are injected into them.
3. **BEV encoder:** queries all camera tokens, queries the aligned previous BEV, then refines the result with a six-layer 2D-RoPE transformer and register tokens.
4. **Unified BEV decoder:** shares spatial upsampling before predicting risk, drivable, distribution, lane class/direction, stop line, and traffic-light state.
5. **Planning decoder:** eight learned mode tokens query BEV features and produce future waypoints, confidence logits, and multi-label behavior logits.

The BEV coordinate system is ego-centric and left-handed: `x` points forward and `y` points right. Always use the configured bounds and raster conversion helpers instead of assuming image-axis direction.

## Data system

### Heterogeneous CARLA collection

CARLA 0.9.15 runs in a Python 3.7 worker; modern encoding, storage, model code, and orchestration run in Python 3.12. Newline-delimited JSON carries control messages, while a named shared-memory arena carries large sensor payloads.

```mermaid
sequenceDiagram
    participant C as Python 3.12 collector
    participant S as Shared-memory arena
    participant W as Python 3.7 CARLA worker
    participant D as Disk
    C->>S: create fixed-capacity arena
    C->>W: init(config, arena_name)
    C->>W: run_scene(seed, weather, route)
    loop every synchronous tick
        W->>W: agent control -> world.tick()
        W->>W: gather every sensor with the same frame_id
        W->>S: append sensor payloads
    end
    W-->>C: return offset/size/shape/dtype index
    C->>S: read frames lazily
    C->>D: encode RGB to H.265; write remaining data to LMDB
```

See [the collector design](data/carla_data_collector/README.md) for protocol, schema, and lifecycle details.

### Scene layout

```text
data/carla_data_collector/dataset/
└── scenes/
    └── scene_000000/
        ├── rgb_front.mp4
        ├── rgb_front_left.mp4
        ├── rgb_front_right.mp4
        └── lmdb/
            ├── data.mdb
            └── lock.mdb
```

The LMDB stores scene metadata, per-frame ego and actor state, camera calibration, depth, semantic labels, optional optical flow, semantic LiDAR, traffic-light state, and route-relevant traffic-control data. RGB remains in per-camera video files.

`SingleFrameSceneBase` builds a lightweight `(scene_dir, frame_idx)` index. Readers and video decoders are created lazily inside DataLoader workers, use a bounded per-worker LRU cache, and close LMDB/video handles on eviction. Batches favor consecutive frames from one scene before shuffling at batch granularity to reduce random H.265 seeks.

### Reconstruction

An independent semantic-LiDAR pipeline fuses static points in CARLA world coordinates and builds local models for vehicles and pedestrians. It produces sparse truncated unsigned distance fields (TUDF) by default and optional Poisson meshes. These artifacts support inspection and scene-representation experiments; they are not inputs to `DrivingModel` and provide no reinforcement-learning signal.

```powershell
python -m data.multiframe_pointcloud_fusion.run
python -m data.mesh_reconstruction.run
python -m data.mesh_reconstruction.run --mesh

python -m vis.reconstructed_pointcloud_vis.run --input 0
python -m vis.reconstructed_udf_vis.run --input 0
python -m vis.reconstructed_mesh_vis.run --input 0
```

## Training

ByteDrive-Toy provides a unified training CLI (`python -m train.run --task <task>`) supporting perception pretraining, end-to-end driving, BEVSeg compression, and continuous-latent VAE representation learning. All tasks are supervised on fixed offline datasets. `clone_loop/` consumes trained checkpoints for simulation inference and evaluation only.

```powershell
# 1. Single-frame perception multi-task (semantics + metric depth)
python -m train.run --task perception

# 2. Unified 4-frame monocular BEV & trajectory flow-matching driving
python -m data.trajectory_vocabulary.run --actions
# Prepare the v10 driving slice cache with data.driving_cache.prepare_driving_cache(samples, cfg).
python -m train.run --task driving

# 3. BEVSeg 64-table grouped discrete codebook compressor (with optional cache warmup)
python -m train.run --task bevseg --prepare-bevseg-cache
python -m train.run --task bevseg

# 4. Privileged continuous Gaussian latent BEV VAE
python -m train.run --task privileged_bev_vae

# 5. Full-BEV 64x spatial compression continuous latent VAE
python -m train.run --task bevseg_vae64
```

### Supported training tasks

| Task (`--task`) | Model & Architecture | Loss / Objective | Input Data & Cache |
| --- | --- | --- | --- |
| `perception` | `PerceptionModel`: frozen DINOv3 + 3-layer trunk + PixelShuffle heads | 29-class CE, symlog metric depth, spatial depth gradient, depth-range BCE | Single-frame RGB (front camera) |
| `driving` | `UnifiedDrivingModel`: 12-layer block-causal visual trunk, 512-word planar acceleration decoder, and BEV flow decoder | One-hot word selection, standardized planar acceleration residual endpoint matching, and BEV flow matching | 4-frame forward monocular RGB; 3-second action/risk slice cache (`data/driving_cache`) |
| `bevseg` | `BEVSegCompressor`: 384-dim residual encoder, 64-codebook tables, ICNR decoder | Weighted semantic reconstruction, lane direction cosine loss, codebook entropy regularization | Single-frame 64m×64m BEVSeg rasters; lossless NumPy cache (`data/bevseg_cache`) |
| `privileged_bev_vae` | `PrivilegedBEVVAE`: 4-stage continuous Gaussian latent encoder-decoder | Semantic reconstruction loss + KL divergence with periodic noise/rank measurement | Single-frame 64m×64m BEVSeg rasters |
| `bevseg_vae64` | `BEVSegVAE64`: 64-channel 4×4 continuous latent VAE (64x spatial compression) | Shared BEVSeg reconstruction loss + KL divergence regularization | Single-frame 64m×64m BEVSeg rasters |

### Default optimizer and runtime settings

| Setting | Default | Description |
| --- | ---: | --- |
| Optimizer | AdamW | Only updates trainable parameters actually used in the active forward path |
| `epochs` | 10 | Target total epoch number, not the count of additional epochs |
| `batch_size` | 32 | Configurable in `config/default.yaml` or via CLI/env overrides |
| `num_workers` | 4 | DataLoader background worker processes |
| `compile` | true on CUDA | In-place PyTorch inductor compilation; eager on CPU |
| `lr` | `1e-4` | Base learning rate |
| `weight_decay` | `1e-5` | AdamW weight decay |
| `resume` | true | Automatically searches and resumes the highest epoch checkpoint in `train/ckpt/<task>/` |

Checkpoints are saved per-task under `train/ckpt/<task>/epoch_XXX.pt`, excluding frozen backbone weights while preserving optimizer states for seamless resume. Use `--prepare-bevseg-cache` with `--task bevseg` to pre-generate missing compressed rasters without constructing model weights.

## Pretrained weights

- Baidu Netdisk: [ByteDrive-Toy pretrained weights](https://pan.baidu.com/s/1Fc8xh40ODsYug3Sc2GMBjQ?pwd=v5pw), extraction code `v5pw`
- Current driving checkpoint: [ByteDrive-Toy Releases](https://github.com/ByteDriveTeam/ByteDrive-Toy/releases), file `driving20260803.pt`

> [!IMPORTANT]
> Older driving weights from the Netdisk are incompatible with the current model. Put `driving20260803.pt` in `train/ckpt/driving/`; do not mistake an older local `driving.pt` for the current release.

> [!NOTE]
> Load `.pt` files only from trusted sources. PyTorch checkpoints are serialized objects, and this project cannot verify third-party downloads or credentials.

Expected layout:

```text
ByteDrive-Toy/
├── model/dinov3-vits16plus-pretrain-lvd1689m/
│   ├── config.json
│   └── model.safetensors
├── train/ckpt/
│   ├── perception/perception.pt
│   └── driving/driving20260803.pt
└── data/map/
    ├── Town02_HD_map.npz
    └── Town05_HD_map.npz
```

## Installation

| Runtime | Purpose | Main dependencies |
| --- | --- | --- |
| Python 3.12 | Encoding, datasets, training, inference, reconstruction, visualization | PyTorch, Transformers, PyYAML, NumPy, OpenCV, LMDB, msgpack, PyAV, Open3D |
| Python 3.7 | CARLA 0.9.15 worker | `carla`, NumPy, Shapely, NetworkX |

Reference development versions are Python 3.12.9, PyTorch 2.12.1, Transformers 5.13.0, PyYAML 6.0.3, NumPy 2.4.6, OpenCV 4.13.0, LMDB 2.2.1, msgpack 1.2.1, PyAV 17.1.0, and Open3D 0.19.0. They are not pinned; select the PyTorch build for your driver and CUDA runtime.

### Python 3.12 environment

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# Install the CUDA-compatible PyTorch build from https://pytorch.org/get-started/locally/ first.
pip install transformers pyyaml numpy opencv-python lmdb msgpack av open3d
```

CPU fallback is supported but DINOv3 and high-resolution BEV decoding are very slow. Full training should use a BF16-capable CUDA GPU.

### Python 3.7 CARLA worker

```powershell
py -3.7 -m venv data/carla_data_collector/py37_venv
.\data\carla_data_collector\py37_venv\Scripts\Activate.ps1
python -m pip install numpy==1.21.6 shapely networkx

# Install the CARLA 0.9.15 wheel or egg supplied with your CARLA distribution.
```

Start a CARLA 0.9.15 server reachable at `127.0.0.1:2000`. The collector accepts only `*_Opt` maps and defaults to `Town02_Opt`. PyAV/FFmpeg must provide `libx265`. Training newly collected data requires enabling the depth and semantic camera modalities used to construct supervision. The driving dataset also requires the matching `data/map/{Town}_HD_map.npz`.

## Quick start

After installing both runtimes and downloading `train/ckpt/driving/driving20260803.pt`, create `config/release20260803.yaml`:

```yaml
driving_vis:
  checkpoint: train/ckpt/driving/driving20260803.pt

clone_loop:
  inference:
    checkpoint: train/ckpt/driving/driving20260803.pt
```

Run offline visualization if a compatible local scene exists:

```powershell
python -m vis.driving_vis.run --env release20260803 `
  --checkpoint train/ckpt/driving/driving20260803.pt
```

Start CARLA and run one closed-loop episode:

```powershell
.\.venv\Scripts\python.exe clone_loop\run.py --env release20260803 --max-episodes 1
```

## End-to-end workflow

For a smoke test from newly collected data, create `config/basic_chain.yaml`:

```yaml
carla_collector:
  cameras:
    modalities:
      rgb: true
      depth: true
      semantic: true
      optical_flow: false

train:
  epochs: 1
  batch_size: 1
  num_workers: 0
  resume: false

pred_vis:
  checkpoint: train/ckpt/perception/epoch_001.pt

driving_vis:
  checkpoint: train/ckpt/driving/epoch_001.pt

clone_loop:
  inference:
    checkpoint: train/ckpt/driving/epoch_001.pt
```

This is a pipeline check, not a recipe for useful driving performance.

1. Validate configuration and the local backbone:

   ```powershell
   python -c "from config import load_config; c=load_config(env='basic_chain'); print(c.train.device, c.model.driving.work_dim)"
   python -c "from model.perception_model import PerceptionModel; from config import load_config; m=PerceptionModel(load_config(env='basic_chain')); print(sum(p.numel() for p in m.parameters()))"
   ```

2. Start CARLA and collect one or two scenes:

   ```powershell
   python data/carla_data_collector/collector/run.py --env basic_chain --max-scenes 2
   ```

3. Inspect a scene:

   ```powershell
   python -m vis.data_vis.run --env basic_chain --scene 0
   ```

4. Train perception:

   ```powershell
   python -m train.run --task perception --env basic_chain
   ```

5. Initialize and train driving:

   ```powershell
   python -m train.run --task driving --env basic_chain `
     --perception-ckpt train/ckpt/perception/epoch_001.pt
   ```

6. Inspect predictions:

   ```powershell
   python -m vis.pred_vis.run --env basic_chain `
     --checkpoint train/ckpt/perception/epoch_001.pt

   python -m vis.driving_vis.run --env basic_chain `
     --checkpoint train/ckpt/driving/epoch_001.pt
   ```

7. Run one closed-loop episode:

   ```powershell
   .\.venv\Scripts\python.exe clone_loop\run.py --env basic_chain --max-episodes 1
   ```

Press `q` to end the current episode while preserving its logs and videos. Results are written under `clone_loop/out/run_<timestamp>/`: stepwise JSONL, a driving video, an inference diagnostics video, and `summary.json`.

## Configuration

`config/default.yaml` is the single source of default values. `config/schema.py` defines types and constraints. `config.load_config()` recursively merges an optional `config/<env>.yaml` override and validates the result.

Example override:

```yaml
train:
  device: cuda
  batch_size: 2
  num_workers: 2
  epochs: 10
  resume: false

model:
  driving:
    freeze_perception: false
```

```powershell
python -m train.run --task driving --env fresh_driving `
  --perception-ckpt train/ckpt/perception/perception.pt
```

When memory is limited, first reduce `train.batch_size` to 1–4, optionally freeze the perception representation, reduce BEV upsampling channels, and reduce register tokens. Camera height and width must remain divisible by the DINO patch size of 16.

## Visualization

### Raw data browser

```powershell
python -m vis.data_vis.run --scene scene_000000
```

Use `Space` to play/pause; arrow keys or `A`, `,`, and `.` to step; `R/D/M/F/V` for RGB, depth, semantic, flow, and BEV/LiDAR layers; `B/S` for dynamic/static boxes; `W` to save a composite; and `Q` or `Esc` to quit.

### Model predictions

```powershell
python -m vis.pred_vis.run --checkpoint train/ckpt/perception/perception.pt
python -m vis.driving_vis.run --checkpoint train/ckpt/driving/driving20260803.pt
```

The perception renderer compares semantic and depth predictions with ground truth. The driving renderer applies sigmoid/argmax as appropriate, draws directed lane tangents, colors stop lines by light state, overlays multimodal trajectories by confidence, and labels native versus legacy traffic-control associations.

### BEVSeg and VAE visualization

```powershell
# Dataset-only or compressor reconstruction & codebook PCA
python -m vis.bevseg_vis.run --scene scene_000000 --mode dataset
python -m vis.bevseg_vis.run --scene scene_000000 --mode compressor --checkpoint train/ckpt/bevseg/epoch_050.pt

# Privileged continuous VAE reconstruction, latent PCA, and detail canvas
python -m vis.privileged_bev_vae_vis.run --checkpoint train/ckpt/privileged_bev_vae/epoch_050.pt

# Measure latent covariance, effective rank, and perturbation noise
python -m tools.privileged_bev_vae_noise.run --checkpoint train/ckpt/privileged_bev_vae/epoch_050.pt
```

### Trajectory, velocity, and driving action vocabulary

```powershell
# Streaming extraction and K-Means clustering of CARLA ego kinematics
python -m data.trajectory_vocabulary.run

# Select 3-second local planar acceleration words by farthest trajectory sampling
python -m data.trajectory_vocabulary.run --actions

# Render the configured vocabulary (coupled action vocabulary takes priority when both exist)
# Coupled trajectory plots use each word's source initial velocity and source trajectory.
python -m vis.trajectory_vocab_vis.run

# Select a specific coupled action artifact or a decomposed geometry/speed artifact
python -m vis.trajectory_vocab_vis.run --vocab-path data/carla_data_collector/dataset/vocabulary/trajectory_action_vocab.npz
python -m vis.trajectory_vocab_vis.run --vocab-path data/carla_data_collector/dataset/vocabulary/trajectory_geometry_vocab.npy
```

## Repository layout

```text
ByteDrive-Toy/
├── config/                         # Default parameters, schema, and environment overrides
├── data/
│   ├── carla_data_collector/       # Python 3.7/3.12 CARLA collection system
│   ├── perception_dataset/         # Single-frame perception dataset
│   ├── driving_dataset/            # Two-frame driving inputs and multitask ground truth
│   ├── driving_targets/            # Trajectory, behavior, and spatial-field targets
│   ├── driving_cache/              # Signed-Log 3-second driving slice cache
│   ├── hd_map/                     # HD-map rasterization and off-road distances
│   ├── bevseg_synthesis/           # BEVSeg ground-truth rasterization
│   ├── bevseg_cache/               # Lossless compressed NumPy raster cache
│   ├── bevseg_dataset/             # Single-frame ego-aligned BEVSeg dataset
│   ├── trajectory_vocabulary/      # Streaming trajectory and velocity vocabulary clustering
│   ├── multiframe_pointcloud_fusion/
│   └── mesh_reconstruction/        # Sparse TUDF and optional Poisson mesh
├── model/
│   ├── dinov3_backbone/            # Frozen local DINOv3 ViT-S+
│   ├── perception_model/           # Shared representation plus semantic/depth heads
│   ├── bev_encoder/                # Camera/history cross-attention and BEV transformer
│   ├── bev_decoder/                # Fields, lanes, and traffic controls
│   ├── trajectory_decoder/         # Multimodal trajectory, confidence, behavior
│   ├── driving_model/              # Complete driving model & UnifiedDrivingModel alias
│   ├── bevseg_compressor/          # Grouped discrete codebook compressor (64 subword tables)
│   ├── bevseg_vae64/               # Full-BEV 64x spatial compression continuous VAE
│   ├── privileged_bev_vae/         # Continuous Gaussian latent privileged BEV VAE
│   └── mamba3/                     # Single-layer pure PyTorch Mamba-3
├── tools/
│   └── privileged_bev_vae_noise/   # Latent covariance, rank, and perturbation measurement
├── train/
│   ├── run.py                      # Unified CLI for perception, driving, bevseg, and VAE tasks
│   ├── loop/                       # Perception and classic driving forward/backward loop
│   ├── unified_driving/            # Signed-Log flow-matching loss and training loop
│   ├── bevseg/                     # Grouped codebook compressor training loop
│   ├── bevseg_vae64/               # 64x spatial compression VAE training adapter
│   ├── privileged_bev_vae/         # Privileged BEV continuous VAE training loop
│   ├── losses/                     # Multitask loss definitions
│   └── optimizer/                  # AdamW and differential LR parameter grouping
├── vis/
│   ├── data_vis/                   # Interactive raw CARLA scene browser
│   ├── pred_vis/                   # Perception prediction vs ground truth
│   ├── driving_vis/                # Driving fields, lanes, traffic, and trajectory viewer
│   ├── bevseg_vis/                 # BEVSeg raster and compressor reconstruction renderer
│   ├── privileged_bev_vae_vis/     # Privileged VAE reconstruction & latent-PCA viewer
│   ├── trajectory_vocab_vis/       # Trajectory and velocity vocabulary static renderer
│   ├── reconstructed_pointcloud_vis/
│   ├── reconstructed_mesh_vis/
│   └── reconstructed_udf_vis/
├── clone_loop/                     # Synchronous CARLA 0.9.15 closed loop
├── Doc/
│   ├── Index.md                    # English file and documentation index
│   ├── Index.zh-CN.md              # Simplified Chinese file and documentation index
│   ├── DevelopmentGuidelines.md    # Development conventions and guidelines
│   ├── 开发规范.md                 # Chinese development conventions
│   ├── BEVSeg.md                   # BEVSeg raster & compressor contract
│   ├── BEVSegVAE64.md              # 64x spatial compression VAE contract
│   └── PrivilegedBEVVAE.md         # Continuous-latent privileged BEV VAE contract
├── site/                           # Project website
├── LICENSE                         # Apache-2.0
├── README.md                       # English primary documentation
└── README.zh-CN.md                 # Complete Simplified Chinese documentation
```

Most modules use a stable package layout:

```text
<module>/
├── __init__.py
├── <module>.py
└── checks/
    ├── __init__.py
    └── <module>_checks.py
```

## Limitations and troubleshooting

### Research limitations

- Training is offline supervised behavior cloning. `clone_loop` is an inference/control/evaluation loop, not a training loop.
- Current data and released weights focus on `Town02_Opt`, one `BehaviorAgent` style, and a narrow traffic/sensor recipe.
- The model uses two frames, three cameras, and LiDAR; it has no long-term memory beyond one aligned historical BEV.
- The controller targets synchronous CARLA 0.9.15 and does not imply real-vehicle readiness.
- There is no official train/validation/test split, data augmentation, learning-rate schedule, metric suite, early stopping, or best-checkpoint selection by default.
- Perception pretraining uses only `front`; driving uses `front/front_left/front_right`.
- Driving supervision depends on CARLA depth, semantics, traffic-light state, and HD maps. Real-data transfer requires new supervision sources.

### Common issues

<details><summary><strong>DINOv3 cannot be found</strong></summary>

The project uses `local_files_only=True`. Verify `model/dinov3-vits16plus-pretrain-lvd1689m/config.json` and `model.safetensors`, or override `model.dinov3_backbone.model_dir`.
</details>

<details><summary><strong>CUDA configuration falls back to CPU</strong></summary>

Install a CUDA-enabled PyTorch build compatible with the local driver. The entry point falls back when `torch.cuda.is_available()` is false.
</details>

<details><summary><strong>The driving dataset reports a missing HD map</strong></summary>

`Town02_Opt` maps to `data/map/Town02_HD_map.npz`. Supply the matching file for other maps or override `data.driving.map_dir` and `map_name_template`.
</details>

<details><summary><strong>Windows DataLoader or VideoCapture failures</strong></summary>

Run the supported `python -m train.run ...` entry point from the repository root. For diagnosis, set `train.num_workers: 0`.
</details>

<details><summary><strong>`--perception-ckpt` appears to load an old driving model</strong></summary>

`train.resume` defaults to true. Disable it for a new experiment; use `--resume <driving-checkpoint.pt>` only when continuing the same run.
</details>

<details><summary><strong>H.265 encoding fails</strong></summary>

Verify that the active PyAV/FFmpeg build provides `libx265`, and inspect `carla_collector.output.video_codec`.
</details>

<details><summary><strong>The closed-loop model stops, drifts, or collides</strong></summary>

First verify checkpoint coverage, input modalities, camera order, temporal spacing, and calibration against training. Then inspect `episode_XXXX.jsonl` and the inference video. If the pipeline is consistent, poor closed-loop performance can still be an expected limitation of the current narrow offline dataset.
</details>

## Development and license

- [File and documentation index](Doc/Index.md)
- [Development guidelines](Doc/DevelopmentGuidelines.md)
- [BEVSeg raster & compressor contract](Doc/BEVSeg.md)
- [Privileged continuous-latent BEV VAE contract](Doc/PrivilegedBEVVAE.md)
- [BEVSeg 64x continuous-latent VAE contract](Doc/BEVSegVAE64.md)
- [CARLA collector design](data/carla_data_collector/README.md)
- [Closed-loop runner design](clone_loop/README.md)
- [Full Simplified Chinese documentation](README.zh-CN.md)

Configurable values must live in `config/`; runtime checks live in adjacent `checks/` directories; source-file comments and docstrings remain Chinese by project convention; and `Doc/Index.md` must be updated whenever files are added or removed.

ByteDrive-Toy is licensed under the [Apache License 2.0](LICENSE).

## Acknowledgements

- [PETR](https://github.com/megvii-research/PETR)
- [CARLA](https://github.com/carla-simulator/carla)
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
- [BEVFusion](https://github.com/mit-han-lab/bevfusion)
