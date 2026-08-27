# ByteDrive File and Documentation Index

**English** · [简体中文](Index.zh-CN.md)

This is the single navigation entry for project documentation and source files. **Update it in the same commit whenever files are added or removed.** Each entry uses `relative path — one-line responsibility`; companion `X_checks.py` files belong to `X.py` and are not listed separately.

## Guidelines and documentation

- [README.md](../README.md) — English primary overview: architecture, data, training, weights, installation, and usage
- [README.zh-CN.md](../README.zh-CN.md) — Complete Simplified Chinese project overview
- [Doc/DevelopmentGuidelines.md](DevelopmentGuidelines.md) — English primary development rules for documentation, comments, config, validation, and code structure
- [Doc/开发规范.md](开发规范.md) — Simplified Chinese development guidelines
- [Doc/Index.md](Index.md) — English primary file and documentation index
- [Doc/Index.zh-CN.md](Index.zh-CN.md) — Simplified Chinese file and documentation index
- [site/index.html](../site/index.html) — English primary project website
- [site/index.zh-CN.html](../site/index.zh-CN.html) — Simplified Chinese project website

## config/ — configuration and validation

- [config/default.yaml](../config/default.yaml) — Default values for every ByteDrive parameter; the single value source
- [config/schema.py](../config/schema.py) — Configuration types and load-time validation; the single constraint source
- [config/__init__.py](../config/__init__.py) — Configuration entry point: read YAML, construct schema, validate, and return config

## data/ — data loading and preprocessing

- [data/__init__.py](../data/__init__.py) — Data package marker; reads config and persisted datasets
- [data/target_encoding/target_encoding.py](../data/target_encoding/target_encoding.py) — Pure target-encoding functions for symlog physical values and depth-range masks
- [data/single_frame_base/single_frame_base.py](../data/single_frame_base/single_frame_base.py) — Shared single-frame dataset base with scene/frame indexing, bounded readers, and RGB normalization
- [data/scene_batch_sampler/scene_batch_sampler.py](../data/scene_batch_sampler/scene_batch_sampler.py) — Scene-aware batch sampler that groups consecutive frames and shuffles batches to reduce random video seeks
- [data/perception_dataset/perception_dataset.py](../data/perception_dataset/perception_dataset.py) — Single-frame perception dataset producing normalized RGB and semantic/depth targets
- [data/driving_targets/driving_targets.py](../data/driving_targets/driving_targets.py) — NumPy/OpenCV driving targets for BEV fields, trajectories, visible occupancy, and eight behavior labels
- [data/hd_map/hd_map.py](../data/hd_map/hd_map.py) — HD-map loading and rasterization for roads, stop lines, and boundary supervision
- [data/driving_dataset/driving_dataset.py](../data/driving_dataset/driving_dataset.py) — Two-frame, three-camera + LiDAR dataset with voxel statistics, temporal transforms, and multitask supervision
- [data/lidar_voxelization/lidar_voxelization.py](../data/lidar_voxelization/lidar_voxelization.py) — CPU-vectorized LiDAR voxel means and population standard deviations of center-relative metric xyz
- [data/multiframe_pointcloud_fusion/__init__.py](../data/multiframe_pointcloud_fusion/__init__.py) — Public API for multi-frame semantic-LiDAR fusion and dynamic-object reconstruction
- [data/multiframe_pointcloud_fusion/multiframe_pointcloud_fusion.py](../data/multiframe_pointcloud_fusion/multiframe_pointcloud_fusion.py) — Static fusion, object-level dynamic reconstruction, scene checkpoints, and batch processing
- [data/multiframe_pointcloud_fusion/run.py](../data/multiframe_pointcloud_fusion/run.py) — CLI for one-scene or recursive multi-frame point-cloud fusion
- [data/mesh_reconstruction/__init__.py](../data/mesh_reconstruction/__init__.py) — Public mesh-reconstruction API with optional watertight repair
- [data/mesh_reconstruction/mesh_reconstruction.py](../data/mesh_reconstruction/mesh_reconstruction.py) — Static-first mesh reconstruction, dynamic donor reuse, checkpoints, and batch processing
- [data/mesh_reconstruction/run.py](../data/mesh_reconstruction/run.py) — Fused point-cloud reconstruction CLI
- [data/mesh_reconstruction/surface/__init__.py](../data/mesh_reconstruction/surface/__init__.py) — Public Poisson-surface and optional watertight-repair API
- [data/mesh_reconstruction/surface/surface.py](../data/mesh_reconstruction/surface/surface.py) — PyTorch support cropping and Open3D Poisson triangle-mesh generation
- [data/mesh_reconstruction/surface/worker.py](../data/mesh_reconstruction/surface/worker.py) — Isolated Poisson subprocess that suppresses Windows crash dialogs and returns one surface result
- [data/mesh_reconstruction/dynamic/__init__.py](../data/mesh_reconstruction/dynamic/__init__.py) — Public dynamic Poisson, donor reuse, and box-fallback API
- [data/mesh_reconstruction/dynamic/dynamic.py](../data/mesh_reconstruction/dynamic/dynamic.py) — Coverage-based dynamic reconstruction with same-object, similar-object, and box fallbacks
- [data/mesh_reconstruction/udf/__init__.py](../data/mesh_reconstruction/udf/__init__.py) — Public sparse TUDF API for the static world and local dynamic objects
- [data/mesh_reconstruction/udf/udf.py](../data/mesh_reconstruction/udf/udf.py) — Sparse regular-tensor construction of static and dynamic truncated unsigned distance fields

### data/carla_data_collector/ — heterogeneous CARLA collection

- [data/carla_data_collector/README.md](../data/carla_data_collector/README.md) — English primary collector architecture, data flow, output schema, and usage
- [data/carla_data_collector/README.zh-CN.md](../data/carla_data_collector/README.zh-CN.md) — Simplified Chinese collector design
- [scene_layout.py](../data/carla_data_collector/scene_layout.py) — Adapted official utility for static/dynamic CARLA map layout extraction

Shared Python 3.7/3.12 layer:

- [common/protocol/protocol.py](../data/carla_data_collector/common/protocol/protocol.py) — JSON-line control protocol, frame index, and semantic-LiDAR dtype definitions
- [common/shm/shm.py](../data/carla_data_collector/common/shm/shm.py) — Named shared-memory arena for zero-copy payload handoff and scene buffering

Python 3.7 worker:

- [worker/main.py](../data/carla_data_collector/worker/main.py) — Worker subprocess entry point driven through stdin/stdout JSON
- [worker/session/session.py](../data/carla_data_collector/worker/session/session.py) — CARLA world/map lifecycle, optimized maps, no-rendering mode, strict synchronization, weather, and seed
- [worker/actors/actors.py](../data/carla_data_collector/worker/actors/actors.py) — Ego, traffic, and pedestrian creation and destruction
- [worker/sensors/sensors.py](../data/carla_data_collector/worker/sensors/sensors.py) — Configurable RGB/depth/semantic/flow cameras, semantic LiDAR, and collision sensor
- [worker/annotations/annotations.py](../data/carla_data_collector/worker/annotations/annotations.py) — Per-frame dynamic and per-scene static semantic bounding boxes
- [worker/traffic_control/traffic_control.py](../data/carla_data_collector/worker/traffic_control/traffic_control.py) — Native traffic-light lane topology, route association, and persistent control ground truth
- [worker/collect/collect.py](../data/carla_data_collector/worker/collect/collect.py) — Strictly synchronized scene collection of sensors, lights, shared-memory payloads, and frame indices
- [worker/model_runtime/model_runtime.py](../data/carla_data_collector/worker/model_runtime/model_runtime.py) — Trajectory-model control with synchronized 10 Hz ground truth and 2 Hz sensors
- [worker/geometry/geometry.py](../data/carla_data_collector/worker/geometry/geometry.py) — CARLA/numeric geometry conversion and camera-intrinsic derivation

Python 3.12 collector:

- [collector/worker_proc/worker_proc.py](../data/carla_data_collector/collector/worker_proc/worker_proc.py) — Control-pipe client that spawns and drives the Python 3.7 worker
- [collector/routes/routes.py](../data/carla_data_collector/collector/routes/routes.py) — Route queue construction, distance filtering, deterministic shuffle, and similarity removal
- [collector/scenarios/scenarios.py](../data/carla_data_collector/collector/scenarios/scenarios.py) — Reproducible per-scene seed and weather selection
- [collector/encode/encode.py](../data/carla_data_collector/collector/encode/encode.py) — Per-camera BGR sequence encoding to H.265 MP4
- [collector/costs/costs.py](../data/carla_data_collector/collector/costs/costs.py) — Offline raw costs for candidates, current/next states, and history using future 10 Hz ground-truth boxes
- [collector/writer/writer.py](../data/carla_data_collector/collector/writer/writer.py) — Non-RGB scene data and independent kinematics timeline writer for LMDB
- [collector/orchestrator/orchestrator.py](../data/carla_data_collector/collector/orchestrator/orchestrator.py) — Expert/model collection loop, closed-loop advancement, segmented storage, and complete cost backfill
- [collector/run.py](../data/carla_data_collector/collector/run.py) — Python 3.12 collection CLI

## model/ — network definitions

- [model/__init__.py](../model/__init__.py) — Network package marker; reads config without defining defaults
- [model/swiglu/swiglu.py](../model/swiglu/swiglu.py) — Reusable SwiGLU activation
- [model/rope_3d/rope_3d.py](../model/rope_3d/rope_3d.py) — FP32 3D rotary position encoding for caller-supplied coordinates
- [model/residual_block/residual_block.py](../model/residual_block/residual_block.py) — 1D/2D/3D RMSNorm, bottleneck residual blocks, and 2D/3D ConvNeXt blocks
- [model/attention/attention.py](../model/attention/attention.py) — Pre-norm cross/self attention with native PyTorch SDPA and optional patch-only 2D RoPE
- [model/dinov3_backbone/dinov3_backbone.py](../model/dinov3_backbone/dinov3_backbone.py) — Frozen eval-only DINOv3 ViT-S+ returning full selected-layer token sequences
- [model/feature_fusion/feature_fusion.py](../model/feature_fusion/feature_fusion.py) — Per-layer RMSNorm, concatenation, and linear projection of DINO sequences
- [model/feature_trunk/feature_trunk.py](../model/feature_trunk/feature_trunk.py) — Three-layer pre-norm transformer over complete DINO token sequences
- [model/pixel_shuffle_upsampler/pixel_shuffle_upsampler.py](../model/pixel_shuffle_upsampler/pixel_shuffle_upsampler.py) — Cascaded 2x PixelShuffle upsampling to input resolution
- [model/perception_head/perception_head.py](../model/perception_head/perception_head.py) — Residual, channel-reduction, and PixelShuffle perception decoder
- [model/perception_model/perception_model.py](../model/perception_model/perception_model.py) — Shared visual encoder with semantic and depth heads
- [model/frustum_encoding/frustum_encoding.py](../model/frustum_encoding/frustum_encoding.py) — Patch-center/corner by depth-sample 3D frustum geometry encoding
- [model/bev_query_embedding/bev_query_embedding.py](../model/bev_query_embedding/bev_query_embedding.py) — Geometry-only xyz BEV query initialization including vertical samples
- [model/lidar_fusion/lidar_fusion.py](../model/lidar_fusion/lidar_fusion.py) — Fourfold local-statistics encoding with visual-conditioned injection into initial BEV queries
- [model/driving_neck/driving_neck.py](../model/driving_neck/driving_neck.py) — Perception/DINO feature fusion, frustum geometry, and 2D residual processing
- [model/bev_encoder/bev_encoder.py](../model/bev_encoder/bev_encoder.py) — Three-camera and historical-BEV fusion followed by a register-token 2D-RoPE transformer
- [model/bev_decoder/__init__.py](../model/bev_decoder/__init__.py) — Public unified BEV decoder API
- [model/bev_decoder/bev_decoder.py](../model/bev_decoder/bev_decoder.py) — Shared upsampling for the three fields, lane geometry, and traffic controls
- [model/bev_upsampler/__init__.py](../model/bev_upsampler/__init__.py) — Public BEV-specific PixelShuffle upsampler API
- [model/bev_upsampler/bev_upsampler.py](../model/bev_upsampler/bev_upsampler.py) — Spatial convolution and activated residual PixelShuffle stages
- [model/trajectory_decoder/trajectory_decoder.py](../model/trajectory_decoder/trajectory_decoder.py) — Conditional multimode planner using eight learned tokens for 10 Hz trajectories
- [model/driving_model/driving_model.py](../model/driving_model/driving_model.py) — Two-frame three-camera + LiDAR model combining image geometry, voxel statistics, and aligned historical BEV

## train/ — training and evaluation

- [train/__init__.py](../train/__init__.py) — Training/optimization/evaluation package marker
- [train/losses/losses.py](../train/losses/losses.py) — Multitask perception, field, lane, traffic-control, trajectory, behavior, and safety losses
- [train/optimizer/optimizer.py](../train/optimizer/optimizer.py) — Optimizer construction for trainable parameters actually used by the task forward path
- [train/loop/loop.py](../train/loop/loop.py) — Perception/driving forward and loss paths, backward pass, gradient clipping, optimizer step, and logging
- [train/run.py](../train/run.py) — Unified task CLI for configuration, model/data/optimizer construction, epochs, and checkpoints

## clone_loop/ — behavior-cloning closed loop

- [clone_loop/README.md](../clone_loop/README.md) — English primary architecture, operation, scoring, and outputs
- [clone_loop/README.zh-CN.md](../clone_loop/README.zh-CN.md) — Simplified Chinese closed-loop documentation
- [clone_loop/__init__.py](../clone_loop/__init__.py) — Package connecting the heterogeneous CARLA worker with main-runtime inference
- [clone_loop/run.py](../clone_loop/run.py) — Closed-loop CLI and episode orchestration entry point
- [clone_loop/protocol/__init__.py](../clone_loop/protocol/__init__.py) — Public control-protocol API
- [clone_loop/protocol/protocol.py](../clone_loop/protocol/protocol.py) — JSON-line protocol between Python 3.7 simulation and main-runtime orchestration
- [clone_loop/shared_frame/__init__.py](../clone_loop/shared_frame/__init__.py) — Public fixed-capacity sensor-sharing API
- [clone_loop/shared_frame/shared_frame.py](../clone_loop/shared_frame/shared_frame.py) — Cross-interpreter RGB/LiDAR buffers avoiding JSON payload copies
- [clone_loop/routes/__init__.py](../clone_loop/routes/__init__.py) — Public route-queue API
- [clone_loop/routes/routes.py](../clone_loop/routes/routes.py) — Reproducible, deduplicated evaluation routes from CARLA recommended spawn points
- [clone_loop/inference/__init__.py](../clone_loop/inference/__init__.py) — Public three-camera + LiDAR inference and trajectory-selection API
- [clone_loop/inference/inference.py](../clone_loop/inference/inference.py) — Checkpoint loading, temporal state, and safety/route-aware trajectory selection
- [clone_loop/control/__init__.py](../clone_loop/control/__init__.py) — Public trajectory-controller API
- [clone_loop/control/control.py](../clone_loop/control/control.py) — Ego-frame trajectory conversion to normalized CARLA steering, throttle, and brake
- [clone_loop/client/__init__.py](../clone_loop/client/__init__.py) — Public Python 3.7 worker-client API
- [clone_loop/client/client.py](../clone_loop/client/client.py) — Worker spawning and synchronous JSON RPC episode control
- [clone_loop/logger/__init__.py](../clone_loop/logger/__init__.py) — Public step-log and summary API
- [clone_loop/logger/logger.py](../clone_loop/logger/logger.py) — Episode JSONL state/control/selection logging and run summary generation
- [clone_loop/recorder/__init__.py](../clone_loop/recorder/__init__.py) — Public driving and inference recorder API
- [clone_loop/recorder/recorder.py](../clone_loop/recorder/recorder.py) — Per-episode camera video and composite online-inference diagnostics encoding
- [clone_loop/orchestrator/__init__.py](../clone_loop/orchestrator/__init__.py) — Public episode-orchestrator API
- [clone_loop/orchestrator/orchestrator.py](../clone_loop/orchestrator/orchestrator.py) — CARLA, shared sensors, model inference, control, evaluation, and logging orchestration

Python 3.7 simulation worker:

- [clone_loop/worker/__init__.py](../clone_loop/worker/__init__.py) — CARLA closed-loop worker package marker
- [clone_loop/worker/run.py](../clone_loop/worker/run.py) — JSON-command worker CLI that advances simulation and writes shared sensors
- [clone_loop/worker/navigation/__init__.py](../clone_loop/worker/navigation/__init__.py) — Public route-progress and local-target API
- [clone_loop/worker/navigation/navigation.py](../clone_loop/worker/navigation/navigation.py) — Global-route progress tracking and ego-frame near-target generation
- [clone_loop/worker/sensors/__init__.py](../clone_loop/worker/sensors/__init__.py) — Public RGB, semantic-LiDAR, collision, and lane-invasion sensor API
- [clone_loop/worker/sensors/sensors.py](../clone_loop/worker/sensors/sensors.py) — Strictly synchronized front camera triplet, LiDAR, and safety-event sensors
- [clone_loop/worker/runtime/__init__.py](../clone_loop/worker/runtime/__init__.py) — Public CARLA world-lifecycle API
- [clone_loop/worker/runtime/runtime.py](../clone_loop/worker/runtime/runtime.py) — CARLA world, traffic, ego, route, and stepwise closed-loop lifecycle

## vis/ — visualization and log rendering

### Reconstructed mesh

- [vis/reconstructed_mesh_vis/__init__.py](../vis/reconstructed_mesh_vis/__init__.py) — Open3D mesh package combining static/dynamic geometry from unified PT files
- [vis/reconstructed_mesh_vis/run.py](../vis/reconstructed_mesh_vis/run.py) — Mesh viewer CLI
- [vis/reconstructed_mesh_vis/reader/__init__.py](../vis/reconstructed_mesh_vis/reader/__init__.py) — Public mesh PT reader and locator API
- [vis/reconstructed_mesh_vis/reader/reader.py](../vis/reconstructed_mesh_vis/reader/reader.py) — Unified static surface, dynamic local-model, and per-frame pose reader
- [vis/reconstructed_mesh_vis/render/__init__.py](../vis/reconstructed_mesh_vis/render/__init__.py) — Public static/dynamic/trajectory renderer API
- [vis/reconstructed_mesh_vis/render/render.py](../vis/reconstructed_mesh_vis/render/render.py) — Open3D mesh clipping, rigid dynamic placement, and actor trajectories
- [vis/reconstructed_mesh_vis/viewer/__init__.py](../vis/reconstructed_mesh_vis/viewer/__init__.py) — Public global/ego-BEV interactive viewer API
- [vis/reconstructed_mesh_vis/viewer/viewer.py](../vis/reconstructed_mesh_vis/viewer/viewer.py) — Playback, view switching, coloring, and screenshots

### Sparse TUDF

- [vis/reconstructed_udf_vis/__init__.py](../vis/reconstructed_udf_vis/__init__.py) — Sparse TUDF reading, per-frame composition, and Open3D visualization
- [vis/reconstructed_udf_vis/run.py](../vis/reconstructed_udf_vis/run.py) — Sparse TUDF viewer CLI
- [vis/reconstructed_udf_vis/reader/__init__.py](../vis/reconstructed_udf_vis/reader/__init__.py) — Public sparse TUDF reader/locator API
- [vis/reconstructed_udf_vis/reader/reader.py](../vis/reconstructed_udf_vis/reader/reader.py) — Unified sparse TUDF PT reader
- [vis/reconstructed_udf_vis/render/__init__.py](../vis/reconstructed_udf_vis/render/__init__.py) — Public TUDF voxel and trajectory renderer API
- [vis/reconstructed_udf_vis/render/render.py](../vis/reconstructed_udf_vis/render/render.py) — Static-world and current-frame dynamic TUDF to Open3D sparse voxels
- [vis/reconstructed_udf_vis/viewer/__init__.py](../vis/reconstructed_udf_vis/viewer/__init__.py) — Public interactive TUDF viewer API
- [vis/reconstructed_udf_vis/viewer/viewer.py](../vis/reconstructed_udf_vis/viewer/viewer.py) — Playback, global/ego BEV, coloring, and screenshots

### Reconstructed point cloud

- [vis/reconstructed_pointcloud_vis/__init__.py](../vis/reconstructed_pointcloud_vis/__init__.py) — Open3D package for unified PT reading, layered coloring, and trajectory browsing
- [vis/reconstructed_pointcloud_vis/run.py](../vis/reconstructed_pointcloud_vis/run.py) — Point-cloud viewer CLI
- [vis/reconstructed_pointcloud_vis/reader/reader.py](../vis/reconstructed_pointcloud_vis/reader/reader.py) — Static map, dynamic object model, and per-frame pose reader
- [vis/reconstructed_pointcloud_vis/render/render.py](../vis/reconstructed_pointcloud_vis/render/render.py) — Layer filtering, downsampling, coloring, and actor trajectories
- [vis/reconstructed_pointcloud_vis/viewer/viewer.py](../vis/reconstructed_pointcloud_vis/viewer/viewer.py) — Global/current BEV, layer, trajectory, color, and screenshot controls

### Raw data

- [vis/data_vis/__init__.py](../vis/data_vis/__init__.py) — Raw dataset visualization package marker
- [vis/data_vis/run.py](../vis/data_vis/run.py) — Configuration, scene location, and interactive-window CLI
- [vis/data_vis/reader/reader.py](../vis/data_vis/reader/reader.py) — Per-scene LMDB/MP4 reader and modality discovery
- [vis/data_vis/geometry/geometry.py](../vis/data_vis/geometry/geometry.py) — NumPy CARLA transforms and 3D-to-2D projection
- [vis/data_vis/palette/palette.py](../vis/data_vis/palette/palette.py) — Vectorized CARLA semantic-label palette
- [vis/data_vis/draw/draw.py](../vis/data_vis/draw/draw.py) — Boxes, depth, semantics, flow, LiDAR/state trajectory BEV, composite panels, and HUD rendering
- [vis/data_vis/viewer/viewer.py](../vis/data_vis/viewer/viewer.py) — OpenCV timeline, playback, layer controls, and screenshots

### Perception predictions

- [vis/pred_vis/__init__.py](../vis/pred_vis/__init__.py) — Perception-checkpoint loading and prediction/ground-truth visualization package
- [vis/pred_vis/render/render.py](../vis/pred_vis/render/render.py) — Colorized semantic/depth prediction and optional ground-truth canvas rendering
- [vis/pred_vis/run.py](../vis/pred_vis/run.py) — Per-frame inference, rendering, and save CLI

### Driving predictions

- [vis/driving_vis/__init__.py](../vis/driving_vis/__init__.py) — Perspective and BEV field/lane/traffic/trajectory comparison package
- [vis/driving_vis/render/__init__.py](../vis/driving_vis/render/__init__.py) — Public mixed-size driving visualization renderer API
- [vis/driving_vis/render/render.py](../vis/driving_vis/render/render.py) — Color and composite rendering for fields, lanes, traffic controls, trajectories, and camera modalities
- [vis/driving_vis/run.py](../vis/driving_vis/run.py) — Three-camera + LiDAR driving visualization CLI
