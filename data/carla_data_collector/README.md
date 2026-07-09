# Carla 合成数据采集模块 — 设计

本目录从 Carla 收集合成驾驶数据。采用**异构双进程**架构：Carla 相关代码跑在 Python 3.7（Carla 0.9.15
仅兼容 3.7），其余处理（H.265 编码、LMDB 写入）跑在 Python 3.12；两进程经**控制管道 + 共享内存**协作。

---

## 1. 需求与落地映射

| # | 需求 | 实现 |
| --- | --- | --- |
| ① | BehaviorAgent 控主车、TrafficManager 控交通流 | `worker/actors.py`：主车挂 BehaviorAgent；车流交 TM 自动驾驶 |
| ② | 行人必须在可行走范围生成 | 生成点取自 `world.get_random_location_from_navigation()`（导航网格）；`WalkerCrowd` 逐 tick 检测到达并重派新目标，使行人全程持续漫游 |
| ③ | 预读地图可达点，两两组合为起/终点，按距离过滤、去重、成队列依次跑且不重复 | `collector/routes.py`：N×N 距离矩阵向量化，保留 min≤d≤max 的有序对，按 `queue_seed` 随机排序 |
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
| ⑬ | 采集全部交通灯状态 | `worker/collect.py`：场景准备时缓存全部交通灯及触发区位置；每个采样帧记录 ID、状态名和 CARLA 原始状态码 |

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

单帧采集顺序：`apply_control(agent.run_step())` → `world.tick()` → `gather(frame_id)` → 碰撞判定 → 按采样间隔写 arena。

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
    collect.py                # 单场景严格同步采集循环 → 写 arena + 帧索引/交通灯状态
    geometry.py               # carla 几何转换 + 相机内参推导
  collector/                  # Py312：编排与处理
    worker_proc.py            # 派生/驱动 worker 子进程的控制管道客户端
    routes.py                 # 可达点→组合→距离过滤→去重→随机队列
    scenarios.py              # 逐场景随机 seed 与天气预设名（记录）
    encode.py                 # RGB→H.265 mp4
    writer.py                 # 每场景独立 LMDB 写入
    orchestrator.py           # 主循环：建队列→驱动→碰撞重试→读 arena→编码+写库
    run.py                    # CLI 入口
  agents/                     # Carla 官方 agents（BehaviorAgent 等），未改动
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
| `meta` | 场景级元数据：scene_id、seed、天气、路线、地图、fps、相机内外参、lidar 外参、静态包围框、交通灯静态位置、视频文件名 |
| `num_frames` | 帧数 |
| `{i}/meta` | 第 i 帧：frame_id、sim_time、主车状态、动态包围框、全部交通灯状态 |
| `{i}/depth/{cam}` | 第 i 帧该相机深度图（解码为米、float32 H×W）；`depth` 开关关闭则无 |
| `{i}/semantic/{cam}` | 第 i 帧该相机语义图（CityScapes 标签、uint8 H×W；与同名 RGB/Depth 像素对齐）；`semantic` 关则无 |
| `{i}/optical_flow/{cam}` | 第 i 帧该相机光流（float32 H×W×2 运动矢量）；`optical_flow` 关则无（默认关） |
| `{i}/lidar` | 第 i 帧语义 Lidar（结构化点：x,y,z,cos_angle,obj_idx,obj_tag，无损）；`lidar.enabled` 关则无 |

> 各模态由 `config` 的 `cameras.modalities`（rgb/depth/semantic/optical_flow）与 `lidar.enabled` 单独开关；
> 关闭即不创建该传感器、不落盘。RGB 关闭则该场景无 mp4（`video_files` 为空）。

数组以 `(dtype, shape, bytes)` msgpack 打包，结构化 dtype 用 descr 保存以无损还原。
`output.lmdb_map_size_gb` 是**单场景 DB 的增长上限**（初始仅 64MB、按需增长，规避 Windows 预占满）。

---

## 6. 关键约定

- **相机**：每个 rig 视角分别配置 `fov` 与位姿；该视角的 RGB / Depth / 语义图强制共享分辨率/FOV/位姿，
  故三者像素天然对齐、共用该视角的一组内外参；场景元数据的 `intrinsics` 以相机名索引；
  RGB 存 BGR 三通道编码进 mp4，Depth 在 arena 也存 BGR 三通道（丢 alpha 省内存）、由 collector 解码为米后入库，
  语义图只取标签所在的 R 通道存单通道 uint8（标签即最终值、无需解码）。内参 `fx=W/(2·tan(fov/2))`。
- **内存**：新增语义图使单帧内存约增 ~10%（每相机 H×W×1 字节）；如需更长场景，调大 `ipc.arena_size_mb`。
- **交通灯**：场景级 `traffic_lights` 保存每盏灯的 `id/transform/trigger_location/trigger_extent`；帧级
  `traffic_light_states` 保存 `id/state/state_code`。状态名为 `red/yellow/green/off/unknown`，对应 CARLA
  原始码 `0/1/2/3/4`。仅实际落盘的采样帧记录状态；HUD 显示全场统计及最近 `vis.traffic_lights.nearest_count` 盏灯。
- **可复现**：`tm.set_random_device_seed(seed)` + Python/np 随机种子，seed 随场景元数据落盘。
- **碰撞**：碰撞传感器置标志；命中即丢弃整场景缓冲、换新 seed 重跑同一路线，队列不前进，超重试上限则跳过该路线。
- **路线过滤**：用起终点直线距离（实际行驶路径由 BehaviorAgent 规划）；如需真实路网里程，可后续接 GlobalRoutePlanner。
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
