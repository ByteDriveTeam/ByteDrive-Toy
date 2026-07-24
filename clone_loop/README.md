# CARLA 0.9.15 行为克隆闭环

该模块把现有 `DrivingModel` 接入 CARLA 同步仿真，形成“观测 → 双帧推理 → 多模态轨迹选择 →
低层控制 → 下一观测”的闭环。架构沿用数据采集器的异构设计：

```
主环境（PyTorch）                         Py37 worker（CARLA 0.9.15）
────────────────────                     ─────────────────────────
路线队列 / episode 编排 ─── JSON ───────► 重载地图、布置交通流
DrivingModel ◄──────── 共享 RGB 帧 ────── 前向相机严格同步
轨迹安全重排
纯追踪 + 速度 PID ─────── JSON control ─► apply_control → world.tick
逐步 JSONL / summary ◄──── 状态元数据 ─── 路线进度、碰撞、压线、终态
```

## 关键口径

- 输入与训练一致：BGR 转 RGB、ImageNet/DINO 归一化；相机内外参与 `DrivingDataset` 顺序一致。
- 时序与训练一致：主环境按 `waypoint_dt_s / fixed_delta_seconds` 缓存对应时距的历史 RGB/位姿；
  历史未满时以当前帧回填且 `previous_valid=0`。
- 导航目标来自 CARLA 全局路线前方固定弧长点，并转换为 ego 左手系 `(x 前, y 右)`。
- 轨迹先按模型置信度排序，再联合风险场、可行驶场和目标方向评分；非有限或明显发散轨迹拒绝执行。
- 低层控制使用纯追踪横向控制和带积分限幅的纵向 PID；模型停车行为可把目标速度门控为零。
- 碰撞、偏航、卡死、步数上限和到达终点均形成明确 episode 终态。

## 环境

默认直接复用数据采集器已有的 Python 3.7 环境：

`data/carla_data_collector/py37_venv/Scripts/python.exe`

主环境使用项目根 `.venv`。CARLA 服务端须已启动，版本为 0.9.15。全部可调项集中在
`config/default.yaml` 的 `clone_loop` 节；机器差异建议写入 `config/<env>.yaml` 覆盖。

## 运行

从仓库根执行：

```powershell
.\.venv\Scripts\python.exe clone_loop\run.py --max-episodes 1
```

指定环境覆盖：

```powershell
.\.venv\Scripts\python.exe clone_loop\run.py --env carla_local
```

每次运行会在 `clone_loop.output.root/run_<时间>/` 下生成：

- `episode_XXXX.jsonl`：每步观测、控制、模式评分、置信度与行为概率；
- `summary.json`：成功率与每条路线的终态、进度、里程、压线等摘要。

闭环默认加载 `train/ckpt/driving/driving.pt`。检查点不存在或非骨干权重覆盖率低于配置阈值时会硬失败，
避免随机模型被误用于车辆控制。
