# Carla 合成数据采集模块 — 设计

[English](README.md) · **简体中文**

本目录从 Carla 收集合成驾驶数据。采用**异构双进程**架构：Carla 相关代码跑在 Python 3.7（Carla 0.9.15
仅兼容 3.7），其余处理（H.265 编码、LMDB 写入）跑在 Python 3.12；两进程经**控制管道 + 共享内存**协作。

### 按场景配置交通流

交通流参数位于 `carla_collector.simulation.traffic_profiles`，按地图保存列表，列表下标与该地图路线队列中的场景对应。每项可独立设置车辆/行人数量、蓝图过滤器、行人奔跑比例、到达半径和 TrafficManager 端口。超出列表长度的路线复用最后一项，便于为大量场景设置统一基线并只覆盖少数场景。

```yaml
carla_collector:
  simulation:
    maps: {Town10HD_Opt: 2}
    traffic_profiles:
      Town10HD_Opt:
        - {num_vehicles: 20, num_walkers: 5, vehicle_filter: vehicle.*, walker_filter: walker.pedestrian.*, walker_running_pct: 0.0, walker_arrival_radius_m: 2.0, tm_port: 8000}
        - {num_vehicles: 80, num_walkers: 30, vehicle_filter: vehicle.*, walker_filter: walker.pedestrian.*, walker_running_pct: 0.2, walker_arrival_radius_m: 2.0, tm_port: 8000}
```

---

## 1. 需求与落地映射

| # | 需求 | 实现 |
| --- | --- | --- |
| ① | BehaviorAgent 控主车、TrafficManager 控交通流 | `worker/actors.py`：主车挂 BehaviorAgent；车流交 TM 自动驾驶 |
| ② | 行人必须在可行走范围生成 | 生成点取自 `world.get_random_location_from_navigation()`（导航网格）；`WalkerCrowd` 逐 tick 检测到达并重派新目标，使行人全程持续漫游 |
| ③ | 按配置逐地图采集；预读当前地图可达点，两两组合为起/终点，按距离过滤、剔除相似路线、成队列依次跑且不重复 | `simulation.maps` 配置地图名与各自场景数；`collector/routes/routes.py` 将 N×N 距离矩阵向量化，保留 min≤d≤max 的有序对；按 `queue_seed` 随机排序后，起终点均落在 `similarity_threshold_m` 邻域内的同向路线只保留一条 |
| ④ | 自动循环；每场景重载地图；seed 每场景随机且记录；碰撞丢弃该场景数据但不跳过组合，换种子重试 | `collector/orchestrator.py` + `worker/session.py`：逐场景 `load_world` 重置；碰撞→丢弃+换 seed 重试、队列不前进（上限 `collision.max_retries_per_route`） |
| ⑤ | 天气随机 | `worker/session.py` 枚举本机 CARLA 内置天气预设；`collector/scenarios.py` 随机选取并记录预设名，经 `run_scene` 下发；worker 据名应用 |
| ⑥ | 明确相机系统：相机数量、各相机 FOV 与外参可配；同一视角各模态共享参数；各传感器模态可单独开关；默认 6 视、H384×W768 | `worker/sensors.py`：每个 rig 视角按 `cameras.modalities` 开关派生 RGB/Depth/语义/光流相机；各视角内参由自身 FOV+分辨率推导；6 视见 `config` |
| ⑥b | 光流相机（与深度同法配置开关） | `sensor.camera.optical_flow`，输出 (H,W,2) float32 运动矢量，逐相机入 LMDB |
| ⑦ | 带语义分割标签的 Lidar（可开关） | `sensor.lidar.ray_cast_semantic`，原始结构化点（含 obj_idx/obj_tag）无损入库；`lidar.enabled` 关闭则不采 |
| ⑧ | RGB→H.265 mp4，其余→LMDB | `collector/encode.py`（libx265）+ `collector/writer.py`（LMDB） |
| ⑨ | 异构：Carla 跑 Py37、其余跑 Py312 | Py312 派生 Py37 worker 子进程；控制走 JSON 行协议，数据走共享内存 |
| ⑩ | 仅内存积累；超阈值强制结束先落盘；单场景上限 1k 帧；场景结束统一处理写入 | 共享内存 arena 即「场景内存缓冲」，容量即阈值（写满→`arena_full` 提前结束）；帧上限 `collection.max_frames_per_scene` |
| ⑪ | 仅 Opt 地图；关闭静态车辆图层（规避已知 API 问题） | `session.py`：仅接受 `*_Opt`；`world.unload_map_layer(carla.MapLayer.ParkedVehicles)` |
| ⑫ | 采集带语义的包围框 | `worker/annotations.py`：动态 actor（逐帧）+ 静态环境物体 level bbs（每场景一次），均带语义标签 |
| ⑬ | 采集交通灯状态与路线相关控制点 | `worker/traffic_control.py`：缓存原生受控车道/停止 waypoint；逐帧按 BehaviorAgent 当前规划选择下一控制点并记录灯色 |
| ⑭ | 运动学与高带宽传感器异频存储 | `collection.kinematics_every_n_ticks` 独立于 `capture_every_n_ticks`；默认运动学 10Hz、图像/Lidar/标注 2Hz，以 `frame_id/sim_time` 对齐 |
| ⑮ | 模型闭环泛化数据 | `ego.controller=model` 时 10Hz 推理并保存全部候选轨迹；控制器只消费 Winner 轨迹，行为概率仅存档；RGB/语义 LiDAR 仍按 2Hz 落盘 |
| ⑯ | 模型原始代价 | 行驶结束后使用未来 10Hz 真实 actor Box 回填当前位置、下一实际状态、全部候选逐航点和历史累计代价；候选几何分批走 Torch GPU（CUDA 不可用或显存不足时回退 CPU）；所有分量非负、无上限、不裁剪、不归一化、不加权 |

---

## 2. 异构架构与数据流

```
Py312 collector (.venv)                         Py37 worker (py37_venv, Carla 0.9.15)
─────────────────────────                       ──────────────────────────────────────
orchestrator
  ├─ 创建并持有共享内存 arena ───────tagname──────► 打开同一 arena（零拷贝）
  ├─ 派生子进程 ──────────────────────────────────► worker/main.py 控制循环
  │   控制面：stdin/stdout 二进制管道，每条一行 UTF-8 JSON
  │     init(config, arena) / query_spawn_points / run_scene(seed,weather,route) / shutdown
  │
  ├─ run_scene ──命令──► 重载 Opt 地图→布景→预热→严格同步采集
  │                       每帧：收齐所有传感器→写入 arena 槽位
  │   ◄──帧索引(JSON)──   返回 status + frames(offset/size/shape/dtype) + 内外参 + 静态bbox
  │
  ├─ 从 arena 读帧（惰性，单帧驻留）→ RGB 编码 mp4 / 深度解码+lidar 还原→LMDB
  └─ 下一条路线
```

- **为何不用 WebAPI/HTTP**：单场景原始 RGB 可达数 GB，走 HTTP 序列化开销与内存峰值过大。共享内存让数据面零拷贝，
  且该 arena 本身即承担「仅内存积累 + 阈值强制落盘」（⑩）。
- **生产/消费不并发**：场景内 worker 只写 arena、场景结束后 collector 只读，故无需跨进程锁。
- **配置单一来源**：根 `config/` 唯一来源；Py312 加载后经 `init` 下发给 worker，**worker 不读 config 文件**。

---

## 3. 严格同步语义

Carla 同步模式（`synchronous_mode=True` + 固定 `fixed_delta_seconds`），逐 `world.tick()` 推进。
worker 为每个传感器开独立队列，回调仅入队；每帧 `gather` 按 `frame_id` **收齐当前帧全部传感器后才推进下一帧**，
陈旧帧丢弃、超前帧报错——杜绝多传感器跨帧错乱。编码/写盘在场景结束后由 Py312 统一做（不拖慢仿真）。

单步采集顺序：`apply_control(agent.run_step())` → `world.tick()` → 碰撞判定 → 分别按运动学与传感器采样间隔写入。
传感器采样点执行 `gather(frame_id)` 严格收齐；运动学无需等待传感器，可按更高频率独立记录。

---

## 4. 文件结构

```
config/                       # 根：全部参数唯一来源（实现文件只读）
  default.yaml  schema.py  __init__.py

data/carla_data_collector/
  common/                     # 两端共享，纯 Python，3.7/3.12 双兼容
    protocol.py               # 控制管道 JSON 协议 + 帧索引/语义Lidar dtype
    shm.py                    # 匿名共享内存 arena + 顺序分配器
  worker/                     # Py37：Carla 采集子进程
    main.py                   # 子进程入口：JSON 控制循环
    session.py                # 世界/地图生命周期：Opt 地图、关停车层、同步、天气、seed
    actors.py                 # 主车+BehaviorAgent、TM 车流、导航网格行人
    sensors.py                # RGB/Depth/语义/光流 按开关共享多相机、语义 Lidar、碰撞；按帧收齐
    annotations.py            # 带语义包围框（动态逐帧 + 静态每场景）
    collect.py                # 单场景异频采集 → 低频传感器 arena + 高频运动学时间轴
    traffic_control.py        # CARLA 原生受控车道/停止点与 Agent 当前规划关联
    geometry.py               # carla 几何转换 + 相机内参推导
  collector/                  # Py312：编排与处理
    worker_proc.py            # 派生/驱动 worker 子进程的控制管道客户端
    routes/                   # 可达点→组合→距离过滤→相似路线剔除→随机队列
    scenarios.py              # 逐场景随机 seed 与天气预设名（记录）
    encode.py                 # RGB→H.265 mp4
    writer.py                 # 每场景独立 LMDB：非 RGB 帧 + 运动学时间轴
    orchestrator.py           # 主循环：建队列→驱动→碰撞重试→读 arena→编码+写库
    run.py                    # CLI 入口
  agents/                     # BehaviorAgent、actor prediction and risk field extensions
  scene_layout.py             # 地图布局工具（官方示例），未改动
```

每个实现文件配同名 `X_checks.py` 承载其运行期校验（按必要性，无前置条件者不强配）；配置校验集中在 `config/schema.py`。

---

## 5. 输出布局

每个场景一个**自包含目录**，互不影响、可独立丢弃/搬移/并行处理：

```
<output.root>/scenes/scene_000000/
  rgb_front.mp4  rgb_front_left.mp4  ... rgb_back.mp4   # RGB → H.265（每相机一个）
  lmdb/                                                  # 本场景的非 RGB 数据
```

单场景 LMDB 键布局（无 scene 前缀）：

| 键 | 内容 |
| --- | --- |
| `meta` | 场景级元数据：scene_id、seed、天气、路线、地图、fps、相机内外参、lidar 外参、静态包围框、交通灯及其受控车道/停止点、视频文件名 |
| `num_frames` | 帧数 |
| `{i}/meta` | 第 i 个低频传感器帧：frame_id、sim_time、主车状态、动态包围框、全部交通灯状态、当前路线下一交通控制点、`navigation_command`，以及可选 `expert` 决策 |
| `{i}/depth/{cam}` | 第 i 帧该相机深度图（解码为米、float32 H×W）；`depth` 开关关闭则无 |
| `{i}/semantic/{cam}` | 第 i 帧该相机语义图（CityScapes 标签、uint8 H×W；与同名 RGB/Depth 像素对齐）；`semantic` 关则无 |
| `{i}/optical_flow/{cam}` | 第 i 帧该相机光流（float32 H×W×2 运动矢量）；`optical_flow` 关则无（默认关） |
| `{i}/lidar` | 第 i 帧语义 Lidar（结构化点：x,y,z,cos_angle,obj_idx,obj_tag，无损）；`lidar.enabled` 关则无 |
| `num_kinematics` | 独立运动学时间轴的记录数 |
| `kinematics/{k}` | 第 k 个运动学采样：frame_id、sim_time、位姿、线速度、线加速度、角速度、完整车辆控制量，以及可选 `expert` 决策 |

BehaviorAgent 的 `expert` 快照记录状态、决策原因、目标速度和加速度、风险预测、规划路线及实际控制。驾驶缓存只提取用于辅助训练的标注，不恢复旧场景数据。

风险走廊使用主车包围盒半宽和 `risk_lateral_margin_m`（默认 0.35 米）。预测中心始终在走廊外的侧方车辆会降低路径风险，避免相邻车道稳定并行时仅因预测不确定性而减速；预测驶入走廊的车辆仍保留完整风险。行人风险不变。

附近车辆还会生成确定性的贝叶斯多意图假设：保持行驶、减速、加速，以及在同向可行驶车道存在时向左或向右变道。近期加速度和相对车道的横向运动更新各意图概率，每个意图保存独立的未来位置均值和协方差。路线风险和前车速度上限按概率融合这些意图，同时保留原单轨迹安全估计；仅由意图假设产生的横穿风险，合计概率至少为 0.5 才会加入必须停车的硬约束；低概率的假设变道不会单独触发紧急制动。`prediction_multi_intent_enabled`、`prediction_intent_map_radius_m`、`prediction_intent_brake_mps2`、`prediction_intent_accelerate_mps2`、`prediction_intent_lane_change_s` 控制该预测。`expert.prediction.actors[*].intent_modes` 保存原始概率与轨迹，不增加行为标签或训练目标。

对于主车路线内的前车，智能体依据当前车身净距、短时贝叶斯位置分布、反应时间和配置的减速度计算纵向安全速度上限。对于预测横穿路线的车辆，还会比较其预测占用时间与主车通过交叉点的时间，并将速度限制在可提前停车的范围内。物理前后位置校验会排除预测轨迹靠近弯曲路线的后车；横穿预测还要求车辆均值向路线靠近，避免并行车辆触发紧急制动。这使减速能早于三秒重叠风险阈值。`front_min_gap_m`、`front_confidence_z`、`front_lookahead_s` 和 `front_emergency_decel_mps2` 控制安全范围，复用现有的状态和原因标注。

当前净距／接近速度检查使用瞬时观测位置和速度，未来均值、协方差及多意图仍使用贝叶斯后验；避免把滤波速度滞后当成已经让行后车的实测追近。`expert.prediction.actors[*].current_position_world_xy` 和 `observed_velocity_world_xy` 保存这两项原始诊断，不增加辅助训练标签。

正常纵向控制规划三秒速度轨迹，同时考虑当前交通速度上限，以及沿原导航路线的红灯、静止前车、横穿车辆和弯道约束。规划器比较受加加速度约束的加速度意图，使用首步加速度限制下发目标速度变化率；即时碰撞检查仍有最高优先级。`longitudinal_horizon_s`、`longitudinal_step_s`、`longitudinal_preview_s` 分别配置规划时域、最大时间网格和诊断预瞄。`control_longitudinal_kp=0.12`、`control_longitudinal_ki=0.02` 配置速度 PID 增益，其时间步长使用仿真固定步长。`expert.longitudinal_plan` 保存选中轨迹的速度、加速度、路线进度、速度上限采样、距离/速度约束、预瞄速度和实际下发目标；它是原始决策记录，不是新增行为标签或辅助训练目标。

低速跟踪时，`control_coast_speed_mps=3.0` 范围内的小幅超速，在停车／弯道约束仍留有足够余量时允许滑行，避免油门与刹车反复切换。停车距离按 `control_coast_decel_mps2=0.25` 的保守拖曳减速度估算；它是车辆与道路相关的配置假设。没有明确停车余量的冲突不使用滑行替代制动，紧迫安全停车保留原有制动能力。`control_launch_throttle_rate=0.6` 限制低速油门每秒增长；车速低于 0.75m/s 时还受 `control_launch_throttle_max=0.15` 限制，避免起步响应延迟期间积累过大油门请求。

指定导航终点加入提前停车约束，剩余距离沿原路线计算，不用直线跨越弯道。`navigation_goal_approach_s=3.0` 控制末段接近速度，`navigation_goal_stop_buffer_m=0.5` 保留终点缓冲，避免队列清空时从巡航速度直接满刹车。无前车、信号灯、冲突和未完成机动的终点低速尾段使用受控滑行／停车；实际障碍制动仍优先。复用 `APPROACH_STOP`／`STOPPING` 与 `comfortable_braking`，不增加类别；`expert.longitudinal_plan.navigation_goal_distance_m`／`navigation_goal_speed_mps`／`navigation_goal_active` 保存原始诊断。

**终点数据只保留在原始日志中。** 驾驶模型不接收终点距离，也不增加“到达”导航类别。缓存生成排除首次明确终点接近决策及其后的窗口，同时排除未来 3 秒轨迹触及该时间的窗口；动作／几何／速度词表在插值前截去同一尾段。高频运动学和传感器帧共同确定截止时间，优先使用控制前的 `decision_sim_time`。没有明确终点诊断的旧场景及普通红灯停车不会因此被丢弃。缓存命名空间哈希包含 `exclude_explicit_navigation_goal_tail_v1` 采样策略；须重新生成缓存以使用该规则，旧缓存文件保留，张量字段与模型检查点结构不变。`prepare_driving_cache` 返回 `skipped_navigation_goal` 计数；未执行全量缓存重建或训练。

附近没有行人时，行人避让检查直接返回，避免基础障碍物检测器把空行人列表解释为“检查所有车辆”。

原始导航路线始终是唯一的目标路线。后车不会触发自动换道或追加路线；只有原始导航明确给出换道指令时，才使用 `LANE_CHANGE` 状态和 `lane_change_safe` 原因。当前路线被静止车辆阻挡时，同时评估左右临时绕行路径、优先同向车道；明确允许的对向借道还需完整返回与视距检查，越过障碍后回到原路线。主车被人为放置到相邻车道时，也可单独触发路径恢复。独立恢复在路口外按车身包围盒触发：侧向偏移达到 min(route_recovery_trigger_m, 车道半宽 − 车身半宽 − 0.2m) 时启动，持续到中心偏移小于 0.2m；左右按 CARLA 坐标系定义。`expert.maneuver_plan` 保存临时路径、目标点、侧向偏差和障碍物 ID；`expert.route_plan` 仍是原始导航队列。原有 `lane_change.cooldown_s` 原始字段保留并写为零，以维持读取兼容性。

主车速度低于 0.5 m/s、在前车后保持停车时，状态仍为 `STOPPING`，原因记为现有的 `follow_vehicle`。仍在行驶并需要立即制动时使用 `emergency_stop`；制动控制不受这项标签调整影响。

该快照还记录弯道限速、车道居中和行人横穿风险。专家风险属于策略推断，`ego` 与 `bboxes` 才是仿真真值。

有效状态编号为 0 `CRUISE`、1 `FOLLOW`、2 `YIELD`、3 `STOPPING`、4 `LANE_CHANGE`、5 `APPROACH_STOP`、6 `CREEP`、7 `AVOID_LEFT`、8 `AVOID_RIGHT`、9 `RECOVER_LEFT`、10 `RECOVER_RIGHT`。原有原因编号 0–13 不变，新增 14 `obstacle_avoid_left`、15 `obstacle_avoid_right`、16 `route_recover_left`、17 `route_recover_right`。

驾驶缓存提供 `annotation_valid`、`annotation_state`、`annotation_reason`、`annotation_probabilities[3]`（预测冲突概率、左侧车道可用性、右侧车道可用性）及逐维有效掩码。车道可用性只检查车道线与相邻驾驶车道，不保证动态交通间隙安全。风险间距、TTC 与专家规划航点仍保留在场景元数据中，但不作为模型辅助预测目标。

`navigation_command` 始终保留原导航五类：0 直行、1 左转、2 右转、3 右变道、4 左变道；`raw_navigation_command` 保存同一原始指令。独立的 `recovery_command` 为 0 无恢复、1 向右恢复、2 向左恢复，可与任意导航指令同时存在。绕障仅作为状态和原因。当前 v16-static-obstacles 缓存保留 v15 的两路输入和独立恢复嵌入；旧场景缺恢复信息时为 0，过渡版 5/6 指令在存在原始指令时拆回两路。旧缓存命名空间保留，原五类导航权重可直接加载，新恢复嵌入需要训练。

### 静态障碍与安全确认

安全路径尚在 0.5 秒连续确认时，不再统一标为预测冲突。恢复意图保持原左／右恢复状态与原因，`expert.maneuver_safety.admission_status=confirming` 表示仍保持当前侧向位置，尚未执行横移；真正阻挡或风险越限仍标为冲突等待，几何／视距限制使用现有舒适制动原因。旧候选不得覆盖当前路径的拒绝结果。

回放 HUD 将确认期显示为 `CONFIRM`，并优先展示当前路径的阻挡者；`front_obstacle_plan` 留存前向停车约束的对象、间距及紧迫性。风险路线先过滤退化的微小起始段，避免后车被错误投影到前方。`route_recovery_m` 是返回段上限；部分恢复按剩余侧偏、车身长度和曲率储备缩短，防止暂停后终点反复后移；退出后车所在车道按完整所需时间检查。

静态物体立即进入避让规划，**不等待车辆持续静止计时，也不额外等待 3 秒**；进入旁侧空间前仍保留连续间隙确认和全部安全检查。支持 `static.prop.*` 道具及本地图报告的 Static、Poles、GuardRail、Fences、Walls 包围框。环境框按空间网格缓存，道具逐帧更新；检查真实高度重叠、定向足迹、双侧交通、整个通过段及返回段，不为静态物体伪造运动意图或增长的贝叶斯协方差。没有可行路径时停车。

`static_obstacles_enabled` 控制接入，`static_obstacle_radius_m=60` 限制查询范围，`static_obstacle_min_height_m=0.08` 排除近似平面物。`static_obstacle_in_lane_enabled` 开启本车道内小幅避让，按 `static_obstacle_margin_m=0.6` 生成偏移；整车还需保留 `static_obstacle_boundary_margin_m=0.15` 道路余量。可行的本车道内方案优先于借道，同向借道优先于对向借道。余量是规划参数，实际跟踪净距另行统计，不能把参数值当作实测保证。

道具复用现有逐帧 bbox 字段及 `semantic=static / source=actor`，原生环境框仍存场景级 `static_bboxes`。训练碰撞风险同时纳入道具与固体环境框，按高度和范围过滤；v16 命名空间隔离旧版忽略静态物的风险标签，旧缓存不删除。普通障碍停车且 `navigation_goal_active=false` 时不会被终点过滤误删。最新驾驶模型输入、输出头、BEV 类别和检查点形状均不改变；没有增加行为状态／原因类别或到达指令。无 RGB 回放不等于完整视觉训练集，仍需重新生成真实训练缓存。

模型模式额外包含：

| 键 | 内容 |
| --- | --- |
| `num_world_states` / `world/{t}` | 10Hz 完整 GT：自车运动、动态 actor 世界系 OBB/速度、全部灯态、路线相关停止线、路线进度、碰撞与压线事件 |
| `num_model_steps` / `model/{t}/meta` | 10Hz 推理元数据：输入/下一帧 ID、Winner、置信度、mode score、行为概率（仅存档）、由 Winner 轨迹得到的控制量 |
| `model/{t}/trajectories` | 全部候选轨迹 `[M,T,2]`，不是只有 Winner |
| `model/{t}/candidate_cost_terms` | 每条候选每个航点的原始代价 `[M,T,K]`；未来环境来自实际执行后采得的 10Hz GT actor Box，灯色保持预测时刻状态 |
| `model/{t}/current_cost_terms` / `next_cost_terms` | 当前所在位置与 Winner 执行一个 tick 后实际状态的原始代价 `[K]` |
| `model/{t}/historical_cost_terms` | 从场景开始逐项累加的实际原始代价 `[K]` |
| 对应 `*_valid` | 逐项有效位；候选无法定义的执行器变化/真实碰撞事件不伪造为监督 |

`meta.cost_terms` 固定记录 K 维名称、类别与单位，覆盖安全性、合规性、交互与博弈能力、通行效率、舒适性与控制质量。
`meta.cost_semantics` 明确声明 lower-is-better、最小 0、最大无上限，以及不裁剪/不归一化/不加权/不保存总分。

> 各模态由 `config` 的 `cameras.modalities`（rgb/depth/semantic/optical_flow）与 `lidar.enabled` 单独开关；
> 关闭即不创建该传感器、不落盘。RGB 关闭则该场景无 mp4（`video_files` 为空）。
> `meta.sensor_dt_s` 与 `meta.kinematics_dt_s` 明确记录两条时间轴的采样间隔；下游通过
> `frame_id` 或 `sim_time` 做精确关联，不应假定二者索引相同。

数组以 `(dtype, shape, bytes)` msgpack 打包，结构化 dtype 用 descr 保存以无损还原。
`output.lmdb_map_size_gb` 是**单场景 DB 的增长上限**（初始仅 64MB、按需增长，规避 Windows 预占满）。
每个场景写入并关闭后会建立 compact 临时副本，逐键逐值校验完全一致，再原子替换 `data.mdb`；Windows 下替换前复制原文件 DACL，键、值、读取结果和访问权限不变，只移除 LMDB 尾部空洞。

---

## 6. 关键约定

- **相机**：每个 rig 视角分别配置 `fov` 与位姿；该视角的 RGB / Depth / 语义图强制共享分辨率/FOV/位姿，
  故三者像素天然对齐、共用该视角的一组内外参；场景元数据的 `intrinsics` 以相机名索引；
  RGB 存 BGR 三通道编码进 mp4，Depth 在 arena 也存 BGR 三通道（丢 alpha 省内存）、由 collector 解码为米后入库，
  语义图只取标签所在的 R 通道存单通道 uint8（标签即最终值、无需解码）。内参 `fx=W/(2·tan(fov/2))`。
- **内存**：新增语义图使单帧内存约增 ~10%（每相机 H×W×1 字节）；如需更长场景，调大 `ipc.arena_size_mb`。
- **交通灯**：场景级 `traffic_lights` 在旧字段 `id/transform/trigger_location/trigger_extent` 之外，新增
  `opendrive_id/pole_index/affected_lane_waypoints/stop_waypoints`。帧级 `traffic_light_states` 保存全部灯的
  `id/state/state_code`；`relevant_traffic_control` 将每盏灯的原生停止点投影到 BehaviorAgent 当前规划，
  选择沿路线最先到达者，并始终写入布尔 `valid`。状态名为 `red/yellow/green/off/unknown`，对应 CARLA
  原始码 `0/1/2/3/4`。下游由 `data.driving.traffic_control.annotation_version` 显式指定解释方式：
  `v1` 忽略已知损坏的原生关联并启用旧 HD Map 几何回退，`v2` 信任该字段及其 `valid=false` 结论。
- **可复现**：`tm.set_random_device_seed(seed)` + Python/np 随机种子，seed 随场景元数据落盘。
- **碰撞**：碰撞传感器置标志；命中即丢弃整场景缓冲、换新 seed 重跑同一路线，队列不前进，超重试上限则跳过该路线。
- **模型失败保留**：模型碰撞不会立即退出；在 `model_collection.collision_followup_steps` 固定窗口内继续正常闭环，
  达到连续无碰撞、回到路线阈值内且相对碰撞点取得规定进度才恢复，否则以 `collision_unrecovered` 保存并终止。
  无合理障碍/红灯原因的低速持续到配置步数时，以 `unjustified_stall` 保存。所有阈值均在默认 YAML 配置。
- **多地图计划**：`carla_collector.simulation.maps` 是“地图名 → 本次场景数”的有序映射。采集器按配置顺序切换地图，
  分别建立路线队列；值为 `0` 时遍历该地图全部合法路线。`--max-scenes N` 仅用于临时调试，会统一覆盖每张地图的数量。
  落盘 `meta.map` 记录实际地图，断点续采也按地图分别维护已采路线。
- **路线过滤**：先用起终点直线距离筛选候选（实际行驶路径由 BehaviorAgent 规划），再按固定随机顺序贪心保留代表路线；
  两条同向路线的起点平面距离和终点平面距离都不超过 `route.similarity_threshold_m` 时视为相似，反向路线不合并，
  配为 `0` 可关闭相似路线剔除。断点续采时，已落盘路线优先作为代表，重复或相似的新候选都会被排除。
  如需按真实路网轨迹比较，可后续接 GlobalRoutePlanner。
- **行人漫游**：每个行人由 `WalkerCrowd` 维护当前目标，行人位置进入 `traffic.walker_arrival_radius_m` 即重派新导航点，
  避免到点后站住不动（采集后半程行人活跃度不衰减）。

---

## 7. 运行

前置：Carla 0.9.15 服务端已启动。依赖已装——py37_venv：carla/numpy/shapely/networkx；.venv：pyyaml/numpy/lmdb/av/msgpack。

从仓库根运行（orchestrator 会自动派生 Py37 worker 子进程，无需手动起服务）：

```
./.venv/Scripts/python.exe data/carla_data_collector/collector/run.py \
    --config config/default.yaml [--max-scenes N]
```

参数可调项全部在 `config/default.yaml`；环境差异用 `config/<env>.yaml` 覆盖（`--env <name>`）。

专家采集保持默认。如只需采集状态（Box、路线/交通状态和运动学），显式关闭 CARLA 渲染：

```yaml
carla_collector:
  simulation:
    no_rendering_mode: true
```

该模式无论相机模态和 LiDAR 的单项开关如何配置，都会跳过全部相机与 LiDAR actor；不能与
`ego.controller: model` 同时使用，因为模型输入需要 RGB 与 LiDAR。

专家采集的失败 attempt 默认丢弃。设置 `carla_collector.collision.save_failed_samples: true` 后，碰撞 attempt
会单独写入 LMDB，并显式记录 `status`、`failed: true` 和 `failure_status`；每个已保存失败样本都会计入场景数。

碰撞后专家采集会继续 `collision.followup_steps` 个 tick（默认 25 帧）。观察窗口结束时若自车速度达到
`collision.recovery_speed_mps` 则恢复采集，否则终止 attempt 并标记为失败。

启用模型泛化采集只需覆盖：

```yaml
carla_collector:
  ego:
    controller: model
```

例如只在 Town01 跑 3 个场景、Town05 跑 8 个场景：

```yaml
carla_collector:
  simulation:
    maps:
      Town01_Opt: 3
      Town05_Opt: 8
```
### 按场景配置交通流

交通流参数位于 `carla_collector.simulation.traffic_profiles`，按地图保存列表，列表下标与该地图路线队列中的场景一一对应。每项可独立设置 `num_vehicles`、`num_walkers`、蓝图过滤器、行人奔跑比例、到达半径和 `tm_port`。当地图的场景数配置为 `0`（遍历全部路线）时，超出列表长度的路线复用最后一项；配置了明确场景数时，列表长度必须覆盖场景数。

```yaml
carla_collector:
  simulation:
    maps: {Town10HD_Opt: 2}
    traffic_profiles:
      Town10HD_Opt:
        - {num_vehicles: 20, num_walkers: 5, vehicle_filter: vehicle.*, walker_filter: walker.pedestrian.*, walker_running_pct: 0.0, walker_arrival_radius_m: 2.0, tm_port: 8000}
        - {num_vehicles: 80, num_walkers: 30, vehicle_filter: vehicle.*, walker_filter: walker.pedestrian.*, walker_running_pct: 0.2, walker_arrival_radius_m: 2.0, tm_port: 8000}
```

连续路径跟踪由 `continuous_route_tracking` 控制，前瞻点沿原导航折线连续移动，减少离散路点切换造成的转向跳变。`turn_alignment_threshold_deg` 在车头尚未对齐时维持弯道限速。

绕障默认要求：车辆连续至少 `obstacle_stationary_time_s=2.0` 秒速度不高于 0.45m/s，在原路线前方 14–32m 且车身与通行走廊重叠；当前导航路点不在路口，也未执行导航换道；不是路线信号灯前的等待，障碍物前方 2–14m 内没有同走廊排队车辆。双侧同时评估，同向安全候选严格优先；同向候选正在等待 0.5 秒确认时，也不抢先选对向。以上距离是沿路线投影的中心间距，不是保险杠净距。

绕障和恢复均检查接收车道、整条路径上的静止车辆，以及未来 `maneuver_prediction_s=3.0` 秒的贝叶斯多意图位置分布。接收车道最小车身净距默认 3m；接近后车或迎面车 TTC 小于 4 秒时不进入该车道，接近速度采用有符号速度计算。多意图按概率加权，冲突概率阈值默认 0.2；安全间隙连续保持 0.5 秒后才执行，中断检查会重置连续计时。执行中每帧复查；绕障等待不继续向障碍物加速，恢复等待期间可沿当前侧向位置缓慢行驶。机动速度上限 `maneuver_speed_mps=2.5`，曲率上限 `maneuver_max_curvature=0.18`；`obstacle_entry_clear_m=7.0` 要求更早完成横移，重规划从实测侧偏／朝向平滑接回路径。

0.5 秒确认用于首次接受机动。恢复已经承诺执行且车身已与接收车道重叠时，当前全部安全检查通过后允许继续完成，不重复冻结半秒；安全检查失败仍暂停。原始诊断 `continuing_committed_merge` 表示这一继续条件，恢复完成后清除承诺。恢复等待的纵向速度按剩余滑行停车余量提前收紧，低速油门随等待速度一起减小，避免临近保留恢复区域时突然刹停。并行车辆到路径的距离使用有限线段，不能把路径端点的切向无限向后延长。

同向绕障同时要求允许进入和返回的车道线条件。`maneuver_allow_opposite=true` 允许有条件临时借用对向车道：两侧分界标记必须均为 Broken，不能跨实线或隔离设施；检查完整通过与返回的车身角点、路段连接和边界。CARLA 对向边界的普通换道标志可能为 NONE，此处只在明确开启借道且双方均为虚线时接受。按完整返回距离估算占道时间，加上 `maneuver_return_buffer_s=4.0`；以 `maneuver_oncoming_speed_mps=15.0` 的设计接近速度预留范围，通过道路切向、坡度和三条碰撞射线检查视距。射线从主车车头侧缘发出，不把观察点放在尚未进入的旁车道中。路口、弯道／坡顶或遮挡导致视距不足时等待。该设计速度是可配置假设，并非对所有来车速度的保证。对向借道获准后安全条件失效时，优先尝试经过复核的回正路径。普通同向车道上实际逆行的车辆也参与完整返回时间检查；预测方向依据实测运动修正，不把逆行速度截成零。

`expert.maneuver_safety` 保存候选侧、阻塞 actor、净距／后车 TTC、预测冲突概率、连续安全时长及等待结果；`expert.maneuver_plan.waiting_for_gap` 与 `proposed_points` 区分当前保持路径和被暂停的拟议路径。未增加状态、原因类别或训练辅助目标；原有 `available_left/right` 仍仅表示静态车道条件。

恢复获准执行时固定回正终点，随后缩短剩余恢复距离；被交通重新阻挡时，等待新间隙后重新确定终点。等待时在导航终点前预留完整回正距离，防止错过目标后兜转。临时路径的曲率同时进入现有弯道限速规划。

车身已经完整进入接收车道后，只忽略同向、侧向稳定且没有明显变道意图的同车道后车追近。车身已部分进入时，若后车留有最小净距且车速不高于主车速度加 1m/s，也允许完成并入，避免已让行后车与主车相互等待。离开当前车道时，满足同样稳定跟随条件、整条路径持续远离且不再次返回其车道的后车也不会冻结恢复；快速后车、逆行、侧向穿越和行人检查仍保留。基础车辆检测返回的距离统一为中心距离，再扣除一次车身长度，避免重复扣除前后车长度导致不必要的紧急制动。纵向控制使用规划器首步加速度；前瞻速度仅用于诊断，避免再次积分导致速度上限失效。

构造场景脚本为 `evaluation/route_recovery_scenarios.py`（采集器 Python 3.7）；用 `evaluation/pack_review_scenes.py`（仓库 Python）将生成的 JSON 封装为标准 scene_*/lmdb。场景保存人工静止障碍物、真值包围盒、原始路线、并行恢复指令、状态和原因，可用 `vis/data_vis/run.py --scene <目录>` 查看；无渲染模式不包含 RGB。

`evaluation/direction_scenarios.py` 在 CARLA 内生成明确车道方向及边线的直路，测试双侧同向、同向优先、对向借道、对向来车、同向车道逆行车和实线拒绝，并恢复原地图。`evaluation/check_maneuver_records.py` 复核碰撞、实际车身角点是否位于可行驶道路、指令／状态契约及真实减速度、横向加速度和 jerk。原始记录及打包 meta 保留 OpenDRIVE，支持离线复核测试道路几何。

舒适性统计同时考虑控制前／后的速度，包含单步降至 0.5m/s 以下或零速的停止帧；不能只统计控制后仍在运动的帧。`run_direction_stage1.py` 冻结控制／测试源码哈希，执行 15 场景、0.7m 净距及实际强减速门槛，再打包并逐帧回读 LMDB、生成预览。

静态道具绕行已完成前一阶段验收；移动冲突避让正在按 `evaluation/MANEUVER_EXTENSION_PLAN.zh-CN.md` 进行本轮构造测试，概率多样性仍等待后续单独授权。

### 固定初始导航与移动冲突避让

每次显式指定目的地／全局计划时，LocalPlanner 封存完整初始路线的值副本。场景级 `initial_navigation` 保存 `points`（世界 XYZ）、`arc_m`、`road_options`、`lane_keys`、与各路线点对应的固定五类 `commands`、生成它们的 `lookahead_m`／`turn_threshold_deg`。行驶时只推进剩余队列；帧级 `navigation_command`／`raw_navigation_command` 从该初始序列按进度读取，不受侧偏、临时绕行或恢复影响。`expert.navigation.assigned_route_index` 记录读取位置。分段采集复用同一份初始导航元数据。`expert.route_plan` 继续保存控制前的完整剩余队列；并行 `recovery_command` 仍独立存在。没有新增驾驶模型输入或导航类别，终点距离仍仅为原始诊断。

`moving_evasion_enabled` 为移动冲突侧向避让开关，受总开关 `obstacle_bypass_enabled` 约束。同车道迎面车辆和预测占用原路线的行人无需等待静止计时；左右候选仍检查边界、车流、概率轨迹及完整返回，返回范围超过静态查询覆盖范围时拒绝入场。额外的 `moving_evasion_horizon_s=6` 时序检查比较减速／维持速度候选，并覆盖 0.8／1.0／1.2 倍行驶进度。`moving_evasion_max_speed_mps=5`、`moving_evasion_max_offset_m=3.5`、`moving_evasion_entry_m=14` 及 `moving_evasion_tracking_reserve_m=0.4` 限制执行；原有横向加速度和曲率限制继续生效。`moving_evasion_hard_mode_probability=0.25` 控制侧向入场的硬意图均值检查；低于它的模式仍参加原有按概率加权的冲突检查，并非忽略。

移动避让分开规划提前进入和越过障碍后的返回站位，获准后跟踪固定临时路径；跟踪误差导致原路径不再可执行时，仅接受重新通过所有检查的平滑修正。使用车辆轴距、转角和转向曲线进行曲率跟踪。它仍是局部执行路径，不调用全局路线重规划。只有全程侧向与时序检查当前获准时，触发对象才由联合转向／速度计划控制；其他对象保留通常的前向制动检查。没有安全候选时制动兜底，不随机强行绕行。

行人候选还比较半车道、四分之三车道的临时偏移；只在相同边界、整车、交通、预测与完整返回检查全部获准时使用，不因目标偏移较小而跳过旁侧安全检查。

已获准的移动避让中，若 PID 仅因小于 1m/s 的跟踪误差给出小于 0.1 的刹车，会额外检验较慢自然减速的完整时序包络；只有所有目标及当前路径仍安全且没有前车／信号约束时才释放该微小刹车。`maneuver_safety.tracking_coast_check` 保存这一证明，安全检查失败仍保留制动。

车辆预测以当前真实位置、速度为起始条件，利用最近一秒位置历史估计加速度，保留滤波协方差和多意图概率，避免把脚本车流速度的逐帧波动当成持续加速。减速外推在停车后停止，已经结束的横移也不会因残留减速度被外推成反向横移；动态整车检查使用两个车体共四条分离轴。行人包含继续、停止、加速与折返假设；`observation_count` 记录历史充分性，近路径首次观测不会被直接当作已知静止目标。行人冲突同时评估受控停车，只有制动轨迹也不安全时才使用紧迫分支；低速滑行须有独立验证的停车余量。

新增内容都是原始诊断：`expert.maneuver_safety.motion_check`／`speed_candidates`、`maneuver_plan.moving_conflict`／`tracking` 和 `expert.pedestrian_braking_plan`。行为状态仍为 11 类、原因仍为 18 类；原有缓存与模型张量、导航五类及恢复三类维持兼容。状态持续时间按最终提交状态计时，避免中间 CRUISE 赋值把连续绕行／恢复时长清零。

新出现的横穿目标或已进入紧迫让行的目标会保持让行决策，不因主车减速后 TTC 风险下降而立即释放：实际位置及有支持的预测轨迹清空原路线走廊持续 0.3 秒后才恢复，当前通过全部检查的侧向避让可接管。长期跟踪目标的远期潜在冲突仍由联合速度／绕行计划处理，不直接强制完全停车。开始让行时取消残留正加速度并松开油门。首次观测仍禁止凭未知运动批准侧向入场；直线制动可使用未来 0.5 秒、行人速度上限 4.5m/s 的可达区域检查，下一帧继续更新。`pedestrian_braking_plan.yield_actor_ids` 和 `corridor_clear_since_sim_time` 便于复核保持／释放的原因。迎面冲突车辆停下后也保留其障碍身份，避免重新等待静止计时期间错误起步。

`evaluation/run_moving_stage3.py` 冻结源码，执行移动冲突与静态／恢复回归，独立复核车体净距、实际道路角点、导航不变性、真实加速度及 jerk，再封装并逐帧回读标准 LMDB。测试为无渲染模式的固定脚本交通流，主车使用 CARLA 真实物理；不代表随机全城压力测试或学习模型推理验收。安全／数据契约门槛与 `comfort_passed` 分开记录：受限行人／双侧受阻停车场景最多保留一次进入速度不超过 2.5m/s 的末段强减速供人工复核，不称为舒适性通过；重复或较高速强减速会拒绝打包。
