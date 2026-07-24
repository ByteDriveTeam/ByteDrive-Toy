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
双路 MP4 ◄─────────────── RGB + 推理输出 ─ 每条路线独立保存
```

## 关键口径

- 输入与训练一致：BGR 转 RGB、ImageNet/DINO 归一化；相机内外参与 `DrivingDataset` 顺序一致。
- 时序与训练一致：主环境按 `waypoint_dt_s / fixed_delta_seconds` 缓存对应时距的历史 RGB/位姿；
  历史未满时以当前帧回填且 `previous_valid=0`。
- 导航目标来自 CARLA 全局路线前方固定弧长点，并转换为 ego 左手系 `(x 前, y 右)`。
- 轨迹先按模型置信度排序，再联合风险场、可行驶场和目标方向评分；非有限或明显发散轨迹拒绝执行。
- 低层控制使用纯追踪横向控制和带积分限幅的纵向 PID；模型停车行为可把目标速度门控为零。
- 碰撞、偏航、步数上限和到达终点形成明确 episode 终态；低速等待不会自动结束。
- Windows 控制台运行期间按 `q` 可手动结束当前 episode 并保存完整日志/录像，然后继续下一条路线；
  其他终端输入 `q` 后按回车。

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

运行期间若模型长期停车或希望提前截断当前片段，按 `q` 即可；不要用 `Ctrl+C`，后者用于结束整个进程。

指定环境覆盖：

```powershell
.\.venv\Scripts\python.exe clone_loop\run.py --env carla_local
```

每次运行会在 `clone_loop.output.root/run_<时间>/` 下生成：

- `episode_XXXX.jsonl`：每步观测、控制、模式评分、置信度与行为概率；
- `episode_XXXX_driving.mp4`：原始前向相机驾驶实况，包含终态帧；
- `episode_XXXX_inference.mp4`：每次模型推理的三行诊断画布，包含相机/HUD、三场、
  道路线、交通控制与全部候选轨迹；
- `summary.json`：成功率与每条路线的终态、进度、里程、压线等摘要。

闭环默认加载 `train/ckpt/driving/driving.pt`。检查点不存在或非骨干权重覆盖率低于配置阈值时会硬失败，
避免随机模型被误用于车辆控制。

横向控制使用纯追踪而非 PID。`clone_loop.control.turn_steer_gain` 默认设为 `0.96`，在平滑前将左右弯
转角等比例缩小 4%，使车辆相对原控制结果略向弯道外侧修正；设为 `1.0` 即恢复原始纯追踪输出。

## 权重含义

`min_weight_coverage` 不是轨迹评分权重。它表示检查点中形状兼容的非 DINO 骨干状态项，占当前模型预期
非骨干状态项的最低比例；默认 `0.95` 会拒绝缺失超过 5% 的残缺或不兼容检查点。

其余四项只参与在线候选轨迹重排，不改变神经网络本身：

```
score =
    confidence_weight × 模型置信度 logit
  - risk_weight × 轨迹沿线平均风险概率
  - drivable_weight × (1 - 轨迹沿线平均可行驶概率)
  - route_alignment_weight × (1 - 轨迹终点方向与导航目标方向的余弦相似度)
```

- `confidence_weight` 越大，越信任模型原始 Mode 排名；
- `risk_weight` 越大，越排斥遮挡、未知或预测危险区域；
- `drivable_weight` 越大，越排斥道路外和可见障碍占用；
- `route_alignment_weight` 越大，越偏向导航目标方向。

默认风险权重为 2，是因为碰撞风险优先级高于路线贴合；但置信度使用未归一化 logit，四项并非天然同尺度，
实际调参应结合 JSONL 中的 `mode_scores` 和推理录像观察，避免某一项长期压倒其余项。
