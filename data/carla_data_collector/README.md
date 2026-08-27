# CARLA Synthetic Data Collector — Design

**English** · [简体中文](README.zh-CN.md)

This directory collects synthetic driving data from CARLA. It uses a **heterogeneous two-process architecture**: CARLA-specific code runs on Python 3.7 because CARLA 0.9.15 targets that runtime, while H.265 encoding, LMDB storage, and orchestration run on Python 3.12. The processes communicate through a **control pipe and shared memory**.

## 1. Requirements and implementation

| # | Requirement | Implementation |
| --- | --- | --- |
| 1 | `BehaviorAgent` controls ego; Traffic Manager controls background traffic | `worker/actors.py` attaches `BehaviorAgent` to ego and enables Traffic Manager autopilot for traffic |
| 2 | Pedestrians spawn only in navigable space | Spawn points come from `world.get_random_location_from_navigation()`; `WalkerCrowd` assigns a new target whenever a walker arrives |
| 3 | Collect configured maps and unique route pairs | `collector/routes/routes.py` vectorizes the NxN distance matrix, filters by distance, removes similar routes, and shuffles deterministically |
| 4 | Automatic scene loop and collision retry | Every scene reloads the map; a collision discards the buffer and retries the same route with a new seed up to the configured limit |
| 5 | Random weather | `collector/scenarios.py` samples and records a locally available CARLA weather preset |
| 6 | Configurable camera rigs and modalities | Every view has configurable FOV and extrinsics; RGB, depth, semantics, and optical flow are independently enabled; the default rig has six `768x384` views |
| 7 | Semantic LiDAR | `sensor.lidar.ray_cast_semantic` is stored losslessly with object IDs and semantic tags |
| 8 | RGB to H.265; remaining modalities to LMDB | `collector/encode.py` uses `libx265`; `collector/writer.py` writes per-scene LMDB data |
| 9 | Python 3.7/3.12 split | The Python 3.12 collector spawns and drives the Python 3.7 worker; JSON carries control and shared memory carries data |
| 10 | In-memory scene buffering with hard limits | A fixed shared-memory arena buffers the scene; capacity or frame limit ends collection before the collector writes the scene |
| 11 | Optimized maps only | `worker/session.py` accepts `*_Opt` maps and unloads `ParkedVehicles` to avoid a known API issue |
| 12 | Semantic bounding boxes | `worker/annotations.py` captures dynamic boxes per frame and static environment boxes per scene |
| 13 | Traffic-control ground truth | `worker/traffic_control.py` records native controlled lanes/stop waypoints and associates the next control with the current expert route |
| 14 | Multi-rate timelines | Kinematics and high-bandwidth sensors use independent intervals, normally 10 Hz and 2 Hz, aligned by `frame_id`/`sim_time` |
| 15 | Model-driven generalization collection | With `ego.controller=model`, inference runs at 10 Hz, all candidate trajectories are retained, and only the winning trajectory drives control |
| 16 | Raw model costs | Future 10 Hz ground-truth actor boxes annotate every candidate waypoint plus current, next, and historical raw costs; values are nonnegative, unbounded, unnormalized, unclipped, and unweighted |

## 2. Architecture and data flow

```text
Python 3.12 collector (.venv)                  Python 3.7 worker (CARLA 0.9.15)
─────────────────────────────                  ─────────────────────────────────
orchestrator
  ├─ create shared-memory arena ─── tagname ─► open the same arena
  ├─ spawn subprocess ───────────────────────► worker/main.py control loop
  │   control plane: binary stdin/stdout, one UTF-8 JSON object per line
  │     init(config, arena) / query_spawn_points / run_scene(...) / shutdown
  │
  ├─ run_scene ──────────────── command ─────► reload map, populate, warm up, tick
  │                                            append synchronized sensors to arena
  │  ◄──────────────────────── frame index ─── status + offset/size/shape/dtype
  │
  ├─ read frames lazily -> encode RGB MP4 / decode depth / write LMDB
  └─ advance route
```

Shared memory avoids serializing several gigabytes of raw RGB through HTTP or another socket protocol. Collection and writing are intentionally sequential: the worker owns the arena while simulating, then the collector owns it while encoding, so no cross-process data lock is needed.

The root `config/` directory is the single source of configuration. Python 3.12 loads and validates it, then sends the materialized configuration to the worker during `init`; the worker never reads config files directly.

## 3. Strict synchronization

The worker enables CARLA synchronous mode with a fixed `fixed_delta_seconds`. Every sensor has its own queue. Callbacks only enqueue data; `gather(frame_id)` waits for all enabled sensors bearing the current frame ID before simulation advances. Stale frames are dropped and future frames are treated as errors.

One collection step is:

```text
apply_control(agent.run_step()) -> world.tick() -> collision check
-> optional kinematics sample -> optional synchronized sensor sample
```

Encoding and persistent storage happen only after the scene ends, so they do not slow simulation ticks.

## 4. Directory structure

```text
config/
  default.yaml  schema.py  __init__.py

data/carla_data_collector/
  common/                     # Pure Python 3.7/3.12 protocol and shared memory
    protocol/                 # JSON commands, responses, frame index, LiDAR dtype
    shm/                      # Named arena and sequential allocator
  worker/                     # Python 3.7 CARLA process
    main/                     # Control-loop entry point
    session/                  # Map/world lifecycle, sync mode, weather, seed
    actors/                   # Ego, BehaviorAgent, traffic, pedestrians
    sensors/                  # Cameras, semantic LiDAR, collision sensor
    annotations/              # Dynamic and static semantic boxes
    traffic_control/          # Native lanes, stop points, route association
    collect/                  # Synchronous multi-rate collection loop
    model_runtime/            # Model trajectory execution in CARLA
    geometry/                 # CARLA/numeric conversions and calibration
  collector/                  # Python 3.12 orchestration and storage
    worker_proc/              # Worker subprocess client
    routes/                   # Route construction, filtering, deduplication
    scenarios/                # Per-scene seed and weather
    encode/                   # RGB to H.265
    costs/                    # Offline raw candidate cost annotation
    writer/                   # Per-scene LMDB
    orchestrator/             # Collection lifecycle and retries
    run.py                    # CLI entry point
  agents/                     # Unmodified CARLA agents
  scene_layout.py             # Adapted official map-layout utility
```

Runtime validation lives in adjacent `checks/` directories. Config validation is centralized in `config/schema.py`.

## 5. Output layout

Every scene is self-contained and may be removed, moved, or processed independently:

```text
<output.root>/scenes/scene_000000/
  rgb_front.mp4  rgb_front_left.mp4  ... rgb_back.mp4
  lmdb/
```

### Core LMDB keys

| Key | Content |
| --- | --- |
| `meta` | Scene ID, seed, weather, route, map, FPS, calibration, static boxes, traffic lights and controlled lanes/stop points, video filenames |
| `num_frames` | Number of low-rate sensor frames |
| `{i}/meta` | Frame ID, simulation time, ego state, dynamic boxes, all light states, and next route-relevant traffic control |
| `{i}/depth/{cam}` | Metric `float32 [H,W]` depth when enabled |
| `{i}/semantic/{cam}` | CityScapes/CARLA `uint8 [H,W]` labels when enabled |
| `{i}/optical_flow/{cam}` | `float32 [H,W,2]` motion vectors when enabled |
| `{i}/lidar` | Lossless structured semantic LiDAR: x, y, z, cosine angle, object ID, semantic tag |
| `num_kinematics` | Number of independent kinematics samples |
| `kinematics/{k}` | Frame ID, time, pose, linear velocity/acceleration, angular velocity, and full vehicle control |

### Additional model-mode keys

| Key | Content |
| --- | --- |
| `num_world_states` / `world/{t}` | Complete 10 Hz ground truth: ego motion, dynamic world-frame OBB/velocity, lights, relevant stop line, route progress, collision and lane-invasion events |
| `num_model_steps` / `model/{t}/meta` | Inference input/next IDs, winner, confidence, mode scores, archived behavior probabilities, and executed control |
| `model/{t}/trajectories` | Every candidate trajectory `[M,T,2]`, not only the winner |
| `model/{t}/candidate_cost_terms` | Raw per-candidate, per-waypoint costs `[M,T,K]` using future ground-truth actors observed after execution |
| `model/{t}/current_cost_terms` / `next_cost_terms` | Raw cost vector at the current and actual next state |
| `model/{t}/historical_cost_terms` | Component-wise accumulated actual cost from scene start |
| matching `*_valid` | Per-term validity; undefined counterfactual or event terms are not fabricated |

`meta.cost_terms` records names, categories, and units. `meta.cost_semantics` declares lower-is-better, minimum zero, no maximum, and no clipping, normalization, weighting, or stored total score.

Modalities are independently controlled by `cameras.modalities` and `lidar.enabled`. Disabled sensors are not created and write no data. `meta.sensor_dt_s` and `meta.kinematics_dt_s` describe distinct timelines; downstream code must join by `frame_id` or `sim_time`, not by assuming identical indices.

Arrays are packed through msgpack as `(dtype, shape, bytes)`. Structured dtype descriptions are retained for exact reconstruction. `output.lmdb_map_size_gb` is the per-scene growth ceiling. After closing a scene, the writer builds a compact copy, verifies every key and value, preserves the Windows DACL, and atomically replaces `data.mdb`.

## 6. Important conventions

- **Cameras:** all modalities for one rig view share resolution, FOV, pose, and therefore pixel alignment. RGB is encoded from BGR, depth is decoded to meters by the collector, and semantics retain only the label channel. `fx = W / (2 * tan(fov/2))`.
- **Memory:** a semantic image adds roughly one byte per pixel per camera. Increase `ipc.arena_size_mb` for longer scenes.
- **Traffic lights:** scene metadata records OpenDRIVE ID, pole index, affected lanes, and stop waypoints. Per-frame state is `red/yellow/green/off/unknown`, corresponding to CARLA codes `0/1/2/3/4`. `relevant_traffic_control` projects native stops onto the current `BehaviorAgent` route and always stores an explicit `valid` flag. Annotation version `v1` uses the legacy HD-map fallback; `v2` trusts native association, including `valid=false`.
- **Reproducibility:** CARLA Traffic Manager, Python, and NumPy use the scene seed, which is stored in metadata.
- **Expert collisions:** the entire scene is discarded and the same route retries with a new seed. The route advances only after success or retry exhaustion.
- **Model failures:** a collision does not stop immediately. Collection continues for `model_collection.collision_followup_steps` and saves `collision_unrecovered` unless the model recovers according to configured conditions. Unexplained prolonged low speed saves `unjustified_stall`.
- **Multiple maps:** `carla_collector.simulation.maps` is an ordered map-name-to-scene-count mapping. Zero means every valid route. `--max-scenes N` is a temporary debug override applied to every map.
- **Route filtering:** straight-line endpoint distance filters candidates; `BehaviorAgent` still plans the driven path. Similar same-direction routes are greedily deduplicated using `route.similarity_threshold_m`. Reverse routes remain distinct.
- **Pedestrians:** `WalkerCrowd` assigns a new navigation point within `traffic.walker_arrival_radius_m`, preventing walkers from becoming stationary late in a scene.

## 7. Running

Start a CARLA 0.9.15 server. Install `carla/numpy/shapely/networkx` in `py37_venv` and `pyyaml/numpy/lmdb/av/msgpack` in the root `.venv`.

From the repository root:

```powershell
.\.venv\Scripts\python.exe data\carla_data_collector\collector\run.py `
  --config config\default.yaml
```

Use `--max-scenes N` for a small debug run and `--env <name>` for `config/<name>.yaml`. The orchestrator starts the Python 3.7 worker automatically.

Expert collection is the default. For state-only collection (boxes, route/traffic state, and kinematics), explicitly disable CARLA rendering:

```yaml
carla_collector:
  simulation:
    no_rendering_mode: true
```

This skips all camera and LiDAR actors regardless of their individual modality switches. It cannot be combined with
`ego.controller: model`, whose policy input requires RGB and LiDAR.

Failed expert attempts are discarded by default. Set `carla_collector.collision.save_failed_samples: true` to write
collision attempts to their own LMDB; each records `status`, `failed: true`, and `failure_status`, and each saved
failure increments the scene count.

After a collision, the expert collector continues for `collision.followup_steps` ticks (25 by default). If the ego
vehicle reaches `collision.recovery_speed_mps`, collection resumes; otherwise the attempt terminates and is marked failed.

Enable model-driven collection with:

```yaml
carla_collector:
  ego:
    controller: model
```

Collect three Town01 scenes and eight Town05 scenes with:

```yaml
carla_collector:
  simulation:
    maps:
      Town01_Opt: 3
      Town05_Opt: 8
```
