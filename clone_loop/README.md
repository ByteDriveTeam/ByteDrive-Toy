# CARLA 0.9.15 Behavior-Cloning Closed Loop

**English** · [简体中文](README.zh-CN.md)

This module connects the existing `DrivingModel` to synchronous CARLA simulation and forms the loop

```text
observation -> two-frame inference -> multimodal trajectory selection
            -> low-level control -> next observation
```

It follows the collector's heterogeneous runtime design:

```text
Main environment (PyTorch)                    Python 3.7 worker (CARLA 0.9.15)
──────────────────────────                    ─────────────────────────────────
Route queue / episode orchestration ─ JSON ─► reload map and populate traffic
DrivingModel ◄── three RGB views + XYZ LiDAR shared regions ─ synchronized sensors
Safety-aware trajectory reranking
World-anchored rolling plan + pure pursuit/PID ─ JSON control ─► apply_control -> world.tick
Step JSONL / summary ◄──────── state metadata ─ progress, collision, lane invasion, terminal state
Two MP4 streams ◄───────────── RGB + inference output ─ one pair per route
```

## Core semantics

- Inputs match training: BGR is converted to RGB and normalized for ImageNet/DINO; semantic LiDAR is encoded in the main runtime as center-relative xyz metric means and population standard deviations in 0.5 m voxels. Camera axes, intrinsics, and extrinsics follow `data.driving.cameras` order.
- Temporal spacing matches training. The main runtime caches historical RGB, LiDAR, and pose at `waypoint_dt_s / fixed_delta_seconds`; before history is available, it reuses the current frame and sets `previous_valid=0`.
- RGB uses fixed shared frames. LiDAR uses a separate variable-length FP32 xyz shared region and transmits the valid point count through the protocol. Capacity overflow fails explicitly instead of truncating points.
- The navigation target is sampled at a fixed arc length ahead on the CARLA global route and transformed into the ego left-handed frame `(x forward, y right)`.
- Trajectories are ordered by model confidence, then jointly scored by risk, drivable probability, and target alignment. Non-finite or clearly divergent candidates are rejected.
- Each winning ego-frame trajectory is anchored in world coordinates. The active plan keeps the previous 0.5 s, blends toward the fresh prediction over the next 0.5 s, and uses the fresh prediction for the remaining horizon.
- Pure pursuit and the integral-limited speed PID consume that same persistent reference. Lookahead grows with speed, contracts with path curvature, and is capped by the path distance inside the stable commitment/blending window.
- Obstacle-stop and red-light-stop behavior probabilities gate target speed through separate enter/release thresholds. A tracking error above 2 m or a discontinuous simulation clock reseeds the plan.
- Collision, route deviation, step limit, and destination arrival produce explicit episode terminal states. Waiting at low speed does not end an episode automatically.
- On Windows, press `q` to end the current episode while preserving complete logs and videos, then continue to the next route. In other terminals, type `q` and press Enter.

## Environment

The default worker reuses the data collector's Python 3.7 environment:

```text
data/carla_data_collector/py37_venv/Scripts/python.exe
```

The main runtime uses the repository-root `.venv`. A CARLA 0.9.15 server must already be running. All configurable values live under `clone_loop` in `config/default.yaml`; machine-specific values should be placed in a `config/<env>.yaml` override.

## Running

From the repository root:

```powershell
.\.venv\Scripts\python.exe clone_loop\run.py --max-episodes 1
```

Press `q` when a model remains stopped or the current segment should be cut short. `Ctrl+C` terminates the entire process.

Use an environment override with:

```powershell
.\.venv\Scripts\python.exe clone_loop\run.py --env carla_local
```

Each run creates `clone_loop.output.root/run_<timestamp>/` containing:

- `episode_XXXX.jsonl`: time-aligned input/next observations, prediction, active reference, tracking diagnostics, and control;
- `episode_XXXX_driving.mp4`: a left/front/right driving view including the terminal frame;
- `episode_XXXX_inference.mp4`: a three-row diagnostic canvas with cameras/HUD, spatial fields, lane and traffic-control predictions, every candidate, and the active reference trajectory;
- `summary.json`: aggregate success rate and per-route terminal state, progress, distance, and lane-invasion information.

By default the runner loads `train/ckpt/driving/driving.pt`. It fails when the checkpoint is absent or compatible non-backbone weight coverage is below the configured threshold, preventing a random or incompatible model from controlling the vehicle.

Lateral control remains pure pursuit rather than steering PID, but its input is the persistent plan transformed back into the current ego frame. `lookahead_min_m/max_m`, `lookahead_time_s`, and `lookahead_curvature_gain` adapt tracking distance; the commitment/blending horizon caps it to stable plan geometry. `behavior_stop_threshold` and `behavior_stop_release_threshold` provide stop hysteresis for configured stop-label indices.

## Weight and scoring semantics

`min_weight_coverage` is not a trajectory score weight. It is the minimum fraction of shape-compatible, non-DINO state entries found in the checkpoint relative to the current model. The default `0.95` rejects checkpoints that omit more than 5% of those entries.

The remaining four weights affect online candidate reranking only; they do not change the network:

```text
score =
    confidence_weight * model confidence logit
  - risk_weight * mean risk probability along the trajectory
  - drivable_weight * (1 - mean drivable probability along the trajectory)
  - route_alignment_weight * (1 - cosine similarity to target direction)
```

- Larger `confidence_weight` trusts the model's original mode ordering more strongly.
- Larger `risk_weight` rejects occluded, unknown, or predicted-dangerous regions more strongly.
- Larger `drivable_weight` rejects off-road and visibly occupied regions more strongly.
- Larger `route_alignment_weight` favors the navigation target direction more strongly.

The default confidence weight is `1.0`; the three auxiliary reranking weights currently default to `0.0`, so the model confidence ordering is unchanged unless an environment override enables those costs. Confidence is an unnormalized logit, so enabled terms do not naturally share a scale; tune them with `mode_scores` and the inference video.

## Research boundary

This is a **behavior-cloning inference and evaluation loop**, not a reinforcement-learning loop. Episode progress, collisions, and terminal states are recorded but never converted into rewards, replay data, gradients, or online policy updates. The current data distribution is narrow, so closed-loop distribution shift and compounding error remain expected limitations.
