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
  agents/                     # CARLA agents with improved BehaviorAgent, forecasting, and risk field
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
| `{i}/meta` | Frame ID, simulation time, ego state, dynamic boxes, all light states, next route-relevant traffic control, `navigation_command`, and optional `expert` decision |
| `{i}/depth/{cam}` | Metric `float32 [H,W]` depth when enabled |
| `{i}/semantic/{cam}` | CityScapes/CARLA `uint8 [H,W]` labels when enabled |
| `{i}/optical_flow/{cam}` | `float32 [H,W,2]` motion vectors when enabled |
| `{i}/lidar` | Lossless structured semantic LiDAR: x, y, z, cosine angle, object ID, semantic tag |
| `num_kinematics` | Number of independent kinematics samples |
| `kinematics/{k}` | Frame ID, time, pose, linear velocity/acceleration, angular velocity, full vehicle control, and optional `expert` decision |

For BehaviorAgent scenes, `expert` records the pre-tick CARLA RoadOption, the full original LocalPlanner waypoint queue (world coordinates, lane IDs, and route distance), driving state and reason IDs, target speed/acceleration, Bayesian actor forecasts and route-risk summaries, applied control, collision/lane-invasion counters, and both decision and result frame IDs/times. `ego.behavior_agent.route_plan_horizon_m` limits only risk evaluation. `SceneReader.expert_decision(i)` and `expert_route_plan(i)` read these fields. The driving cache packs selected expert annotations without reconstructing legacy scenes.

The agent matches CARLA's native light stop waypoints to the queued route's road and lane before braking. It caps speed ahead of tight turns using route curvature (`corner_lateral_accel_mps2`, `corner_preview_m`) and holds a moderate speed until heading and lane position recover (`turn_alignment_speed_mps`, `lane_centering_speed_mps`). `planner_base_min_distance_m` and `planner_distance_ratio` keep the tracking waypoint closer on tight bends. Pedestrian forecasts derive velocity from consecutive simulator positions because walker actor velocity can lag actual motion; a predicted crossing within stopping distance triggers early braking. The active reason IDs for `curve_speed` and `lane_centering` are 12 and 13.

The risk corridor uses the ego vehicle's bounding-box half-width plus `risk_lateral_margin_m` (0.35 m by default). A vehicle whose predicted mean stays outside this corridor has reduced route risk, so stable traffic in an adjacent lane does not cause braking from forecast uncertainty alone. A vehicle predicted to enter the corridor keeps full risk. Pedestrian risk is unchanged.

Nearby vehicles also receive deterministic Bayesian intent hypotheses: keep course, brake, accelerate, and, where same-direction driving lanes exist, change left or right. Recent acceleration and lane-relative motion update the probabilities; each hypothesis has separate future means and covariances. Route risk and front speed caps weight the hypotheses while retaining the original single-track safety estimate. A mode-only crossing adds a hard stop constraint only when its combined probability is at least 0.5; low-probability lane changes do not trigger emergency braking on their own. `prediction_multi_intent_enabled`, `prediction_intent_map_radius_m`, `prediction_intent_brake_mps2`, `prediction_intent_accelerate_mps2`, and `prediction_intent_lane_change_s` configure this forecast. `expert.prediction.actors[*].intent_modes` stores its raw probabilities and trajectories; no behavior label or training target is added.

For a vehicle already in the ego route, the agent also computes a longitudinal speed cap from its current physical gap, near-term Bayesian position distribution, reaction time, and configured deceleration. For a vehicle predicted to cross the route, it checks whether the vehicle's predicted occupancy overlaps the time when the ego would traverse the crossing, then caps speed to stop before it. A physical forward-position check excludes rear vehicles whose extrapolated tracks happen to approach a curved route; crossing forecasts also require the vehicle mean to move toward the route, preventing parallel traffic from triggering emergency braking. This begins slowing before the three-second overlap risk threshold is reached. `front_min_gap_m`, `front_confidence_z`, `front_lookahead_s`, and `front_emergency_decel_mps2` control the safety envelope. It reuses existing state and reason labels.

Normal longitudinal control plans a three-second speed trace using traffic caps and route-relative light, stopped-vehicle, crossing-vehicle and curve limits. It evaluates jerk-limited acceleration intentions and applies the first acceleration to the issued target; immediate collision checks retain priority. `longitudinal_horizon_s`, `longitudinal_step_s`, and `longitudinal_preview_s` configure the horizon, maximum grid interval and diagnostic preview. Speed PID gains are `control_longitudinal_kp=0.12`, `control_longitudinal_ki=0.02`, using the simulator's fixed time step. `expert.longitudinal_plan` stores samples, constraints, preview and issued targets as raw metadata without adding training targets.

Below `control_coast_speed_mps=3`, small tracking overshoots can coast only when explicit stop/curve constraints retain sufficient room, using the configurable drag assumption `control_coast_decel_mps2=0.25`. Unbounded conflicts cannot coast instead of braking; immediate safety stops retain their braking authority. `control_launch_throttle_rate=0.6` limits low-speed throttle rise per second; below 0.75m/s, `control_launch_throttle_max=0.15` prevents a delayed launch from accumulating excessive throttle demand.

Assigned navigation goals receive an anticipatory stop constraint using distance along the original route. `navigation_goal_approach_s=3` and `navigation_goal_stop_buffer_m=0.5` taper the final approach instead of waiting for an empty queue to issue full braking. A low-speed terminal coast/stop requires no lead vehicle, signal, conflict or unfinished maneuver; actual hazards retain priority. Existing `APPROACH_STOP`/`STOPPING` and `comfortable_braking` labels are reused. Raw goal distance and speed are stored in `expert.longitudinal_plan`.

When no pedestrian is nearby, the pedestrian avoidance check returns immediately. This prevents the base obstacle detector from treating an empty pedestrian list as a request to inspect every vehicle.

The assigned navigation route remains authoritative. Rear traffic cannot initiate a lane change or append a route. `LANE_CHANGE` and `lane_change_safe` require an explicit navigation lane-change option; `lane_change.cooldown_s` remains zero for compatibility. A stationary blocker can trigger temporary bypass after comparing both sides, preferring same-direction lanes; enabled opposing-lane borrowing also requires full-return and visibility checks. The path returns to the original waypoint queue. Displaced-ego recovery starts outside junctions before the body crosses the edge, at min(route_recovery_trigger_m, lane half-width minus body half-width minus 0.2 m), with 0.2 m centering hysteresis. `expert.maneuver_plan` records the temporary path; `expert.route_plan` remains the assigned route.

When ego speed is below 0.5 m/s behind a close vehicle, the agent retains `STOPPING` but records `follow_vehicle` as the reason. `emergency_stop` is reserved for a vehicle that must brake immediately while moving. The control used to hold the stopped vehicle is unchanged.

Active state IDs are 0 `CRUISE`, 1 `FOLLOW`, 2 `YIELD`, 3 `STOPPING`, 4 `LANE_CHANGE`, 5 `APPROACH_STOP`, 6 `CREEP`, 7 `AVOID_LEFT`, 8 `AVOID_RIGHT`, 9 `RECOVER_LEFT`, and 10 `RECOVER_RIGHT`. Reason IDs 0–13 retain their prior meanings; 14 `obstacle_avoid_left`, 15 `obstacle_avoid_right`, 16 `route_recover_left`, and 17 `route_recover_right` are appended.

| `expert` field | Content |
| --- | --- |
| `decision_frame_id/result_frame_id`, `decision_sim_time/result_sim_time`, `decision_ego_transform` | Pre-control and post-tick alignment; world pose `[x,y,z,roll,pitch,yaw]` |
| `navigation`, `route_plan.points` | Raw RoadOption and target waypoint; complete planner queue with world pose, road/lane IDs, junction flag, and `arc_m` |
| `maneuver_plan` | Temporary bypass or recovery path, world-space target, original-route offset, and blocking actor ID |
| `state/state_id`, `reason/reason_code`, `state_since_sim_time/state_duration_s` | Expert state machine and decision reason, including approach, stop, and low-speed creep states |
| `target_speed_mps/target_acceleration_mps2/speed_limit_mps`, `applied_control` | Expert target values and issued throttle, steer, brake, and direction controls |
| `prediction.actors`, `risk` | Up to 16 neighboring actor forecasts with mean world trajectories, covariance, times, and extents; route risk and estimated conflict time |
| `traffic_light_id/traffic_light_distance_m`, `lane_change`, `events` | Relevant light, lane-change availability/cooldown, cumulative collision/lane-invasion counts |

`ego` and `bboxes` remain simulator ground truth. Forecasts, risks, and states under `expert` are policy-produced decision records, not ground-truth future trajectories. The driving cache exposes `annotation_state`, `annotation_reason`, `annotation_probabilities[3]` (forecast conflict, left lane availability, right lane availability), and their validity masks. Lane availability checks markings and adjacent driving lanes; it does not certify a safe traffic gap. Expert route points, risk gaps, and TTC remain in scene metadata but are not auxiliary model targets.

`navigation_command` retains the five original route IDs (0 straight, 1 left turn, 2 right turn, 3 right lane change, 4 left lane change); `raw_navigation_command` preserves the same instruction. The parallel `recovery_command` channel uses 0 none, 1 recover right, 2 recover left and can coexist with any route instruction. Bypass appears only in state/reason labels. The v16-static-obstacles cache retains v15's two inputs and separate recovery embeddings. Legacy scenes default to no recovery; interim commands 5/6 are split only when raw_navigation_command is available. Existing five-row route weights load directly; new recovery embeddings require training.

### Static solids and maneuver confirmation

Clear-path admission confirmation is no longer labeled as a predicted conflict. The existing recovery state/reason remains active, while `expert.maneuver_safety.admission_status=confirming` explicitly means lateral motion is still held. Real actor conflicts retain their wait; geometry/sight rejection uses the existing comfortable-braking reason. Current checks take precedence over stale candidate diagnostics.

The HUD shows `CONFIRM` separately and reports current-path blockers. `front_obstacle_plan` preserves the front stopping object, gap and urgency. Degenerate tiny route-origin segments are removed before risk projection so a rear actor cannot acquire a false front station. `route_recovery_m` caps the return length; partial returns shrink with remaining offset, body length and curvature reserve instead of restarting at the full length after a pause. Source-lane rear followers are checked against the complete lane-exit time.

Static objects enter planning immediately, without the moving-vehicle stationary timer or an extra three-second delay. Receiving-space admission still requires continuous clearance and all safety checks. Supported inputs are spawned `static.prop.*` actors and native Static, Poles, GuardRail, Fences and Walls boxes. Native geometry is spatially indexed once; props update each tick. Height overlap and oriented footprints constrain the entire pass and return, including traffic on both sides. Static solids have no invented motion intent or growing forecast covariance. Safe in-lane offsets take priority, followed by same-direction borrowing and conditional opposing borrowing; otherwise ego stops.

Configuration exposes `static_obstacles_enabled`, a 60 m query radius, 0.08 m minimum height, `static_obstacle_in_lane_enabled`, 0.6 m planned obstacle margin and 0.15 m planned road-boundary margin. Actual tracking clearance is audited separately. Props reuse current per-frame bbox fields with `semantic=static / source=actor`; native boxes remain scene metadata. Collision risk supervision now includes solid static boxes and height filtering. The v16 namespace preserves old caches while preventing reuse of labels that ignored statics. An explicit `navigation_goal_active=false` keeps ordinary obstacle stops eligible. Model inputs, output heads, BEV classes and checkpoint shapes are unchanged; no new state/reason classes or arrival command are added. Review scenes without RGB are not a full visual training dataset.

### Additional model-mode keys

Navigation-end stopping remains in raw recordings only. The driving model receives no goal distance and no new arrival command. `expert.longitudinal_plan.navigation_goal_active` identifies terminal approach; cache preparation excludes that decision and the following tail, plus windows whose future target reaches it. Action, geometry and speed vocabulary sampling truncates the same tail before interpolation. High-rate kinematics and sensor-frame diagnostics are checked together, using pre-control `decision_sim_time` where available. Legacy scenes without explicit terminal diagnostics and ordinary traffic-light stops retain their previous eligibility. The cache namespace hash now includes `exclude_explicit_navigation_goal_tail_v1`: regenerate the cache to apply this selection policy. Old files remain intact; tensor fields and model checkpoint shapes do not change. Preparation reports `skipped_navigation_goal`. No full cache rebuild or training has been performed for this change.

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

Continuous route tracking (`continuous_route_tracking`) interpolates the lookahead point along the assigned route; `turn_alignment_threshold_deg` keeps the turn speed cap until the heading aligns. Constructed scenarios can be recorded with `evaluation/route_recovery_scenarios.py` using collector Python, then converted to standard SceneReader scenes with `evaluation/pack_review_scenes.py` using repository Python. No-rendering recordings contain metadata and geometry without RGB.

Bypass admission requires a vehicle below 0.45 m/s for at least 2 seconds, 14–32 m ahead along the assigned route with lateral footprint overlap, outside junction and navigation lane-change phases, without a route signal stop or queued vehicle 2–14 m ahead of the blocker. Both sides are evaluated together, with strict preference for safe same-direction candidates, including while their clearance timer settles. These are projected center distances.

Bypass and recovery check receiving-lane body gaps (default 3 m), signed closing TTC (4 s), parked vehicles along the entire path, and 3-second Bayesian multimodal forecasts (probability threshold 0.2). An opening must remain safe for 0.5 s; interrupted evaluation resets that timer. Checks repeat every tick. Bypass waiting stops consuming entry space; recovery can continue along its held offset. Maneuver speed is capped by `maneuver_speed_mps=2.5`, curvature by `maneuver_max_curvature=0.18`. `obstacle_entry_clear_m=7` moves lateral completion earlier; replanning joins measured offset and heading smoothly. `expert.maneuver_safety` records both candidates, direction, geometry, cost, blockers, TTC and clearance duration. State/reason vocabularies and auxiliary targets are unchanged; `available_left/right` retain their static semantics.

Current gaps use measured position and velocity; future means, covariance and intent modes retain the Bayesian posterior. Raw fields `expert.prediction.actors[*].current_position_world_xy` and `observed_velocity_world_xy` expose the observations. When a complete recovery path moves away from a stable current-lane follower with sufficient gap, freezing recovery for diffuse uncertainty is avoided; fast followers, inward cuts, opposing traffic and pedestrians retain their checks.

The 0.5-second admission timer applies to initial entry. A committed recovery already overlapping its receiving lane may continue as soon as every current safety check passes; unsafe paths remain paused. `continuing_committed_merge` records this condition and commitment resets after completion. Waiting recovery reserves a coasting tail and reduces creep throttle with its speed limit. Parallel clearance uses finite path segments, so a backward extension of a return tangent cannot create a fictitious conflict.

Same-direction bypass requires marking permission for both entry and return. `maneuver_allow_opposite=true` admits temporary opposing-lane borrowing only across borders marked Broken on both sides. CARLA ordinary lane-change flags may be NONE there; explicit borrowing plus both Broken marks is required. The entire pass and return must fit the driving corridor. Return time plus `maneuver_return_buffer_s=4` determines the reservation, using `maneuver_oncoming_speed_mps=15` as the configurable approaching-speed assumption. Native road continuity, direction/grade limits and three collision rays must establish sufficient sight; rays originate at the ego's front side edge, not inside the unentered adjacent lane. Junctions, curves, crests and occlusion can reject entry. This design speed is an assumption, not a guarantee about all approaching vehicles. Revoked reservations trigger an attempt at a validated return. Observed wrong-way traffic in nominally co-directed lanes also receives the full-return check; intent prediction preserves measured motion direction.

An accepted recovery fixes its return endpoint and reduces remaining return distance on later ticks. A new traffic blockage resets that endpoint for the next accepted opening. Waiting reserves enough route length before the navigation goal for the full return. Temporary-path curvature enters the existing turn-speed planner.

Once inside the receiving lane, only stable co-directed followers without strong lane-change intent are ignored during centering; wrong-way, lateral crossing and pedestrian checks remain active. Detector distances are normalized before body lengths are subtracted once. Longitudinal commands use the optimizer's first acceleration rather than differentiating an already integrated preview speed.

`evaluation/direction_scenarios.py` generates explicit straight-road direction/border tests in CARLA and restores the prior map. Records and scene metadata retain OpenDRIVE for offline road-footprint auditing. `check_maneuver_records.py` checks collisions, road corners, decision contracts and measured acceleration/jerk. Static-prop bypass, active moving-conflict/pedestrian evasion and probabilistic choice remain separate stages in `evaluation/MANEUVER_EXTENSION_PLAN.zh-CN.md`. Pedestrian evasion must check both sides and the entire return against all supported obstacle types.

Comfort auditing includes ticks decelerating below 0.5m/s or to zero, using speed before and after control. `run_direction_stage1.py` freezes source hashes, runs 15 cases, enforces 0.7m body clearance and a measured strong-deceleration gate, then packs and reads every LMDB frame back before rendering previews.

### Immutable initial navigation and moving-conflict evasion

An explicit global plan assignment captures a detached full initial route. Scene metadata `initial_navigation` stores world-XYZ `points`, `arc_m`, `road_options`, `lane_keys`, fixed per-waypoint five-class `commands`, and their `lookahead_m`/`turn_threshold_deg`. Frame navigation reads this sequence by progress; `expert.navigation.assigned_route_index` exposes the cursor. Detours only change local execution. `expert.route_plan` remains the full pre-control remaining queue, and `recovery_command` stays independent. Segments of one drive retain the same initial navigation. Model inputs, label vocabularies, checkpoints and tensor shapes are unchanged; goal distance remains raw diagnostics only.

`moving_evasion_enabled` adds immediate oncoming-in-lane/pedestrian candidate evaluation, subject to `obstacle_bypass_enabled`. Both sides, road geometry, full return, signed traffic gaps and weighted Bayesian forecasts remain mandatory. Additional six-second admission checks compare speed profiles with 0.8/1.0/1.2 progress envelopes. Limits are configurable through `moving_evasion_max_speed_mps=5`, `moving_evasion_max_offset_m=3.5`, `moving_evasion_entry_m=14` and `moving_evasion_tracking_reserve_m=0.4`; existing curvature/lateral-acceleration limits still apply. `moving_evasion_hard_mode_probability=0.25` controls hard mean-trajectory guards; lower-probability hypotheses still participate in weighted conflict checks.

The escape entry and return station are planned separately. Admitted world-space paths use curvature tracking based on measured wheel geometry; a smooth correction must pass fresh safety checks before replacing one. The trigger actor is controlled by the joint steering/speed proof only while that proof remains admitted; other actors retain ordinary longitudinal guards. Unsafe lateral choices retain braking fallback. Controlled pedestrian stopping is evaluated before an urgent brake, and coasting requires verified stopping room. Vehicle acceleration uses recent position history to avoid false sustained acceleration from oscillating physics velocities. Pedestrian continue/stop/faster/reverse alternatives and observation counts retain uncertainty at first sight.

Raw diagnostics include `maneuver_safety.motion_check`/`speed_candidates`, `maneuver_plan.moving_conflict`/`tracking`, and `pedestrian_braking_plan`; state/reason classes remain 11/18. Final committed states determine exported state duration. `evaluation/run_moving_stage3.py` freezes sources, records moving hazards and prior regressions, audits body/road/navigation/kinematics, then packs and decodes standard LMDB scenes. These are no-rendering scripted-traffic tests with physical ego dynamics, not random full-town stress or learned-policy acceptance. Probabilistic diversity remains a separate, later approval stage.

Yielding for newly appearing crossing targets or an established urgent stop persists until observed and supported predicted footprints clear the original corridor for 0.3 seconds, independently of ego-speed-induced TTC changes. An admitted safe evasion may take over. Established tracks retain joint speed/escape planning: a distant possible conflict alone does not force a complete stop. Yielding cancels stale positive acceleration and throttle. A first observation cannot certify lateral admission; braking may instead pass a 0.5-second reachable-set check with a 4.5m/s pedestrian bound and must be rechecked next tick. `pedestrian_braking_plan.yield_actor_ids` and `corridor_clear_since_sim_time` explain retention and release. A threatening oncoming vehicle retains its obstacle identity after stopping, avoiding a spurious launch during a new stationary timer.

Pedestrian candidates also include half- and three-quarter-lane offsets, with the same body, traffic, probability and full-return checks. Review gates distinguish safety/data contracts from `comfort_passed`. Constrained pedestrian and both-sides-blocked stops may retain one measured terminal strong-deceleration episode entered at no more than 2.5m/s; this is an explicit comfort limitation. Repeated or faster strong deceleration fails packing. Open-side pedestrian evasion, other moving escapes and prior regressions require zero strong deceleration.

During admitted moving evasion, a PID brake below 0.1 for a speed-tracking error below 1m/s may be released only if a fresh slower natural-deceleration envelope clears every actor and no front-vehicle or signal constraint exists. `maneuver_safety.tracking_coast_check` retains this evidence; failed checks keep braking.
