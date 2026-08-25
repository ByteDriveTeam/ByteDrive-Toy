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
Pure pursuit + speed PID ───── JSON control ─► apply_control -> world.tick
Step JSONL / summary ◄──────── state metadata ─ progress, collision, lane invasion, terminal state
Two MP4 streams ◄───────────── RGB + inference output ─ one pair per route
```

## Core semantics

- Inputs match training: BGR is converted to RGB and normalized for ImageNet/DINO; semantic LiDAR is encoded in the main runtime as center-relative xyz metric means and population standard deviations in 0.5 m voxels. Camera axes, intrinsics, and extrinsics follow `data.driving.cameras` order.
- Temporal spacing matches training. The main runtime caches historical RGB, LiDAR, and pose at `waypoint_dt_s / fixed_delta_seconds`; before history is available, it reuses the current frame and sets `previous_valid=0`.
- RGB uses fixed shared frames. LiDAR uses a separate variable-length FP32 xyz shared region and transmits the valid point count through the protocol. Capacity overflow fails explicitly instead of truncating points.
- The navigation target is sampled at a fixed arc length ahead on the CARLA global route and transformed into the ego left-handed frame `(x forward, y right)`.
- Trajectories are ordered by model confidence, then jointly scored by risk, drivable probability, and target alignment. Non-finite or clearly divergent candidates are rejected.
- Low-level control uses pure pursuit laterally and an integral-limited PID longitudinally. A predicted stop behavior may gate target speed to zero.
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

- `episode_XXXX.jsonl`: per-step observation, control, mode score, confidence, and behavior probability;
- `episode_XXXX_driving.mp4`: a left/front/right driving view including the terminal frame;
- `episode_XXXX_inference.mp4`: a three-row diagnostic canvas with cameras/HUD, spatial fields, lane and traffic-control predictions, and every candidate trajectory;
- `summary.json`: aggregate success rate and per-route terminal state, progress, distance, and lane-invasion information.

By default the runner loads `train/ckpt/driving/driving.pt`. It fails when the checkpoint is absent or compatible non-backbone weight coverage is below the configured threshold, preventing a random or incompatible model from controlling the vehicle.

Lateral control uses pure pursuit, not PID. `clone_loop.control.turn_steer_gain` defaults to `0.96`, reducing left/right turn steering by 4% before smoothing so the result tracks slightly farther toward the outside of a curve. Set it to `1.0` to restore the unscaled pure-pursuit command.

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

The default risk weight is 2 because collision risk is prioritized over route alignment. Confidence uses an unnormalized logit, however, so the terms do not naturally share a scale. Tune them with `mode_scores` in the JSONL and the inference video, and check that one term does not dominate persistently.

## Research boundary

This is a **behavior-cloning inference and evaluation loop**, not a reinforcement-learning loop. Episode progress, collisions, and terminal states are recorded but never converted into rewards, replay data, gradients, or online policy updates. The current data distribution is narrow, so closed-loop distribution shift and compounding error remain expected limitations.
