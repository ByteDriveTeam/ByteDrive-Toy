# ByteDrive 文档索引

全项目文档与源文件的单一导航入口。**每次增删文件，同一提交内更新本表。**
每行格式：`相对路径 — 一句话职责`（职责应与该文件文件头首行一致）。
校验伴随文件 `X_checks.py` 附属于 `X.py`，**不单独列入索引**。

## 规范与文档

- [README.md](../README.md) — ByteDrive-Toy 项目总览：代码库、模型架构、数据系统、训练策略、权重下载与运行指南
- [Doc/开发规范.md](开发规范.md) — 项目强制开发规范（文档/注释/配置/校验/简洁）
- [Doc/Index.md](Index.md) — 本文档索引

## config/ — 配置与校验（参数唯一来源）

- [config/default.yaml](../config/default.yaml) — ByteDrive 全部参数默认值（唯一数据来源）
- [config/schema.py](../config/schema.py) — 配置的类型定义与加载期校验（参数约束的唯一来源）
- [config/__init__.py](../config/__init__.py) — 配置加载入口：读 yaml → 构造 schema → 校验 → 返回配置对象

## data/ — 数据读取与预处理

- [data/__init__.py](../data/__init__.py) — 数据读取与预处理包标识：只读消费 config 与已落盘数据集
- [data/target_encoding/target_encoding.py](../data/target_encoding/target_encoding.py) — 监督目标编码：Symlog 物理量、深度范围掩码的纯函数
- [data/single_frame_base/single_frame_base.py](../data/single_frame_base/single_frame_base.py) — 单帧场景数据集共享基类：场景/帧索引、有界 SceneReader 缓存、RGB 归一化（感知与驾驶数据集复用）
- [data/scene_batch_sampler/scene_batch_sampler.py](../data/scene_batch_sampler/scene_batch_sampler.py) — 场景感知批采样器：连续帧同批、批间随机，减少视频随机 seek 与跨场景解码器切换
- [data/perception_dataset/perception_dataset.py](../data/perception_dataset/perception_dataset.py) — 感知模型单帧数据集：把落盘场景逐帧展开，产出归一化 RGB 与语义/深度监督目标（采用所有帧）
- [data/driving_targets/driving_targets.py](../data/driving_targets/driving_targets.py) — 驾驶监督目标编码（numpy/OpenCV）：BEV/轨迹/三场、可见运动占用及八类多标签行为。
- [data/hd_map/hd_map.py](../data/hd_map/hd_map.py) — HD 地图：加载车道折线与交通灯触发区，生成道路、停止线及越界监督。
- [data/driving_dataset/driving_dataset.py](../data/driving_dataset/driving_dataset.py) — 驾驶模型双帧数据集：产当前/上一帧输入、帧间刚性变换、道路线图与驾驶多任务监督。

### data/carla_data_collector/ — Carla 合成数据采集（Py37 worker + Py312 collector 异构）

- [data/carla_data_collector/README.md](../data/carla_data_collector/README.md) — 本采集模块的设计文档（架构/数据流/输出布局/运行）
- [scene_layout.py](../data/carla_data_collector/scene_layout.py) — 从 Carla 地图提取静态/动态场景布局（官方示例改造）

共享层 `common/`（两端纯 Python，3.7/3.12 双兼容）

- [common/protocol/protocol.py](../data/carla_data_collector/common/protocol/protocol.py) — 控制管道 JSON 行命令/响应协议与帧索引/语义Lidar dtype 定义
- [common/shm/shm.py](../data/carla_data_collector/common/shm/shm.py) — 匿名共享内存 arena：跨进程零拷贝传大块数据，兼作场景内存缓冲

Py37 采集端 `worker/`（仅 py37_venv 运行）

- [worker/main.py](../data/carla_data_collector/worker/main.py) — Py37 worker 子进程入口：经 stdin/stdout JSON 协议受 collector 驱动采集
- [worker/session/session.py](../data/carla_data_collector/worker/session/session.py) — Carla 世界/地图生命周期：连接、加载 Opt 地图、严格同步、天气与种子
- [worker/actors/actors.py](../data/carla_data_collector/worker/actors/actors.py) — 主车、交通流与行人的生成与销毁
- [worker/sensors/sensors.py](../data/carla_data_collector/worker/sensors/sensors.py) — 传感器阵列：逐视角按开关创建 RGB/Depth/语义/光流相机、语义分割 Lidar、碰撞传感器
- [worker/annotations/annotations.py](../data/carla_data_collector/worker/annotations/annotations.py) — 带语义的包围框抽取：动态 actor（逐帧）与静态环境物体（每场景）
- [worker/collect/collect.py](../data/carla_data_collector/worker/collect/collect.py) — 单场景严格同步采集循环：逐帧收齐传感器、交通灯状态、共享内存数据与帧索引
- [worker/geometry/geometry.py](../data/carla_data_collector/worker/geometry/geometry.py) — carla 几何对象与纯数值的转换，及相机内参推导

Py312 编排处理端 `collector/`（根 .venv 运行）

- [collector/worker_proc/worker_proc.py](../data/carla_data_collector/collector/worker_proc/worker_proc.py) — 派生并驱动 Py37 worker 子进程的控制管道客户端
- [collector/routes/routes.py](../data/carla_data_collector/collector/routes/routes.py) — 由可达点构建路线队列：两两组合、按直线距离过滤、随机排序、确保不重复
- [collector/scenarios/scenarios.py](../data/carla_data_collector/collector/scenarios/scenarios.py) — 逐场景随机：种子与天气预设（决策与记录都在 collector 侧，便于复现）
- [collector/encode/encode.py](../data/carla_data_collector/collector/encode/encode.py) — 把单相机的 BGR 帧序列编码为 H.265 mp4
- [collector/writer/writer.py](../data/carla_data_collector/collector/writer/writer.py) — 把场景的非 RGB 数据写入 LMDB（深度/语义Lidar/包围框/主车状态/元数据/视频引用）
- [collector/orchestrator/orchestrator.py](../data/carla_data_collector/collector/orchestrator/orchestrator.py) — 采集主循环：建队列→驱动 worker→碰撞重试→读共享内存→编码+写 LMDB
- [collector/run.py](../data/carla_data_collector/collector/run.py) — Py312 采集入口 CLI：加载配置并启动采集主循环

## model/ — 网络结构定义

- [model/__init__.py](../model/__init__.py) — 网络结构定义包标识：只读消费 config，不含可调参数默认值
- [model/swiglu/swiglu.py](../model/swiglu/swiglu.py) — 通用 SwiGLU 激活模块（沿维度二等分为 value/gate）
- [model/rope_3d/rope_3d.py](../model/rope_3d/rope_3d.py) — 通用 3D RoPE 旋转位置编码（只消费调用方传入的三维坐标，全程 FP32）
- [model/residual_block/residual_block.py](../model/residual_block/residual_block.py) — 视觉编码器残差卷积模块（1D/2D/3D RMSNorm、瓶颈残差块与 2D/3D ConvNeXt 块）
- [model/attention/attention.py](../model/attention/attention.py) — Pre-Norm 交叉/自注意力块（多头，PyTorch 原生 SDPA，可选 patch-only 2D RoPE）。
- [model/dinov3_backbone/dinov3_backbone.py](../model/dinov3_backbone/dinov3_backbone.py) — DINOv3 ViT-S+ 视觉骨干：全程冻结 + eval，逐帧输出选定层的完整 Token 序列。
- [model/feature_fusion/feature_fusion.py](../model/feature_fusion/feature_fusion.py) — DINO 多层序列融合：对选定层逐层 RMSNorm 后沿末维拼接，再线性降到预测主干工作维。
- [model/feature_trunk/feature_trunk.py](../model/feature_trunk/feature_trunk.py) — 预测特征主干：完整继承 DINOv3 Token 序列，经三层带 patch-only 2D RoPE 的 Pre-Norm Transformer。
- [model/pixel_shuffle_upsampler/pixel_shuffle_upsampler.py](../model/pixel_shuffle_upsampler/pixel_shuffle_upsampler.py) — 级联像素洗牌上采样：把低分辨率特征逐级 2× 放大回原分辨率
- [model/perception_head/perception_head.py](../model/perception_head/perception_head.py) — 感知解码头：2D 残差块 + 通道压缩 + 级联像素洗牌上采样至原分辨率
- [model/perception_model/perception_model.py](../model/perception_model/perception_model.py) — 共享视觉特征编码器，以及在其上追加语义/深度双头的多任务单帧感知模型。
- [model/frustum_encoding/frustum_encoding.py](../model/frustum_encoding/frustum_encoding.py) — 深度 frustum 位置编码：每 patch 中心+四角×深度采样的候选 3D 坐标 → 逐 patch 几何特征
- [model/bev_query_embedding/bev_query_embedding.py](../model/bev_query_embedding/bev_query_embedding.py) — BEV 查询几何嵌入：仅把 BEV 栅格中心 xyz（含垂直 z 采样）编码为初始查询网格
- [model/driving_neck/driving_neck.py](../model/driving_neck/driving_neck.py) — 驾驶前端 neck：感知 trunk+DINO 原始特征 RMSNorm 融合 + frustum 几何编码 + 2D 残差
- [model/bev_encoder/bev_encoder.py](../model/bev_encoder/bev_encoder.py) — BEV 编码器：融合当前图像与历史 BEV，再由带无位置寄存器的六层二维 RoPE Transformer 提炼
- [model/field_decoder/field_decoder.py](../model/field_decoder/field_decoder.py) — 三场解码头：BEV 特征上采样解码为风险/可行驶/轨迹分布场
- [model/lane_map_decoder/lane_map_decoder.py](../model/lane_map_decoder/lane_map_decoder.py) — 道路细线解码器：共享高分辨率特征输出道路线、相关停止线与交通灯状态。
- [model/trajectory_decoder/trajectory_decoder.py](../model/trajectory_decoder/trajectory_decoder.py) — 条件化多 Mode 规划解码器：以 8 个可学习 Token 查询主感知第 3/6 层特征并回归基线残差。
- [model/driving_model/driving_model.py](../model/driving_model/driving_model.py) — 双帧开环驾驶模型：融合刚性对齐的历史 BEV，解码三场、道路线、交通控制与驾驶输出。

## train/ — 训练 / 评估循环

- [train/__init__.py](../train/__init__.py) — 训练 / 优化 / 评估循环包标识：只读消费 config
- [train/losses/losses.py](../train/losses/losses.py) — 多任务监督损失：感知、驾驶场、道路线、交通控制、轨迹行为及安全约束。
- [train/optimizer/optimizer.py](../train/optimizer/optimizer.py) — 优化器构造：仅优化任务前向实际使用的可训练参数，冻结或未参与前向的模块不纳入。
- [train/loop/loop.py](../train/loop/loop.py) — 训练与评估循环：感知与驾驶两条前向/损失路径，反向 → 梯度裁剪 → 步进并聚合日志
- [train/run.py](../train/run.py) — 训练入口 CLI：按 --task 选择感知/驾驶目标，加载配置 → 建模型/数据/优化器 → 逐 epoch 训练并保存权重。

## clone_loop/ — 行为克隆闭环

- [clone_loop/README.md](../clone_loop/README.md) — CARLA 0.9.15 异构行为克隆闭环的架构、运行与输出说明
- [clone_loop/__init__.py](../clone_loop/__init__.py) — CARLA 0.9.15 行为克隆闭环包：连接异构仿真 worker 与主环境模型推理
- [clone_loop/run.py](../clone_loop/run.py) — CARLA 0.9.15 行为克隆闭环 CLI：加载配置并启动异构 episode 编排
- [clone_loop/protocol/__init__.py](../clone_loop/protocol/__init__.py) — 闭环控制管道协议的公开 API 重导出入口
- [clone_loop/protocol/protocol.py](../clone_loop/protocol/protocol.py) — 定义 Py37 仿真 worker 与主环境闭环编排器之间的 JSON 行协议
- [clone_loop/shared_frame/__init__.py](../clone_loop/shared_frame/__init__.py) — 单帧共享内存区的公开 API 重导出入口
- [clone_loop/shared_frame/shared_frame.py](../clone_loop/shared_frame/shared_frame.py) — 跨解释器复用固定大小 RGB 缓冲，避免每个闭环步经 JSON 复制图像
- [clone_loop/routes/__init__.py](../clone_loop/routes/__init__.py) — 闭环路线队列构造的公开 API 重导出入口
- [clone_loop/routes/routes.py](../clone_loop/routes/routes.py) — 由 CARLA 推荐生成点构造可复现、无重复的闭环评测路线队列
- [clone_loop/inference/__init__.py](../clone_loop/inference/__init__.py) — 闭环模型推理与轨迹选择的公开 API 重导出入口
- [clone_loop/inference/inference.py](../clone_loop/inference/inference.py) — 加载驾驶权重、维护双帧状态，并按置信度/安全场/路线一致性选择闭环轨迹
- [clone_loop/control/__init__.py](../clone_loop/control/__init__.py) — 轨迹跟踪控制器的公开 API 重导出入口
- [clone_loop/control/control.py](../clone_loop/control/control.py) — 把模型选中的 ego 系轨迹转换为 CARLA 归一化转向、油门与制动
- [clone_loop/client/__init__.py](../clone_loop/client/__init__.py) — Py37 worker 子进程客户端的公开 API 重导出入口
- [clone_loop/client/client.py](../clone_loop/client/client.py) — 派生 Py37 CARLA worker，并以同步 JSON RPC 驱动闭环 episode
- [clone_loop/logger/__init__.py](../clone_loop/logger/__init__.py) — 闭环逐步日志与汇总写入器的公开 API 重导出入口
- [clone_loop/logger/logger.py](../clone_loop/logger/logger.py) — 把每个 episode 的闭环状态、控制与选择结果写为 JSONL，并生成运行汇总
- [clone_loop/recorder/__init__.py](../clone_loop/recorder/__init__.py) — 闭环驾驶与逐帧推理录像器的公开 API 重导出入口
- [clone_loop/recorder/recorder.py](../clone_loop/recorder/recorder.py) — 逐 episode 编码前向驾驶实况，并合成模型全部在线推理输出的诊断录像
- [clone_loop/orchestrator/__init__.py](../clone_loop/orchestrator/__init__.py) — 闭环 episode 编排器的公开 API 重导出入口
- [clone_loop/orchestrator/orchestrator.py](../clone_loop/orchestrator/orchestrator.py) — 串联 Py37 CARLA、共享 RGB、驾驶模型、轨迹控制与逐 episode 评测日志

Py37 仿真端 `worker/`

- [clone_loop/worker/__init__.py](../clone_loop/worker/__init__.py) — Py37 CARLA 闭环 worker 包标识
- [clone_loop/worker/run.py](../clone_loop/worker/run.py) — Py37 CARLA 闭环 worker CLI：接收 JSON 命令、推进仿真并把 RGB 写入共享帧区
- [clone_loop/worker/navigation/__init__.py](../clone_loop/worker/navigation/__init__.py) — CARLA 路线进度与局部目标模块的公开 API 重导出入口
- [clone_loop/worker/navigation/navigation.py](../clone_loop/worker/navigation/navigation.py) — 在 CARLA 全局路线中跟踪主车进度，并生成模型所需的 ego 系近端目标
- [clone_loop/worker/sensors/__init__.py](../clone_loop/worker/sensors/__init__.py) — 闭环 RGB、碰撞与压线传感器的公开 API 重导出入口
- [clone_loop/worker/sensors/sensors.py](../clone_loop/worker/sensors/sensors.py) — 创建闭环前向 RGB 与安全事件传感器，并按仿真帧严格同步取图
- [clone_loop/worker/runtime/__init__.py](../clone_loop/worker/runtime/__init__.py) — CARLA 闭环世界生命周期的公开 API 重导出入口
- [clone_loop/worker/runtime/runtime.py](../clone_loop/worker/runtime/runtime.py) — 管理 CARLA 世界、交通流、主车、路线和逐步闭环推进

## vis/ — 可视化与日志渲染

- [vis/data_vis/__init__.py](../vis/data_vis/__init__.py) — 可视化包标识：只读消费采集数据集并渲染
- [vis/data_vis/run.py](../vis/data_vis/run.py) — 可视化入口 CLI：加载配置、定位场景目录、启动交互窗口
- [vis/data_vis/reader/reader.py](../vis/data_vis/reader/reader.py) — 场景读取器：合并单场景的 LMDB 与 mp4 为逐帧数据，探测各模态可用性
- [vis/data_vis/geometry/geometry.py](../vis/data_vis/geometry/geometry.py) — 纯 numpy 复刻 CARLA 坐标变换与 3D->2D 投影
- [vis/data_vis/palette/palette.py](../vis/data_vis/palette/palette.py) — CARLA 语义标签到颜色的调色板与向量化映射
- [vis/data_vis/draw/draw.py](../vis/data_vis/draw/draw.py) — 渲染：3D 框投影、深度/语义/光流着色、lidar+框 鸟瞰图、多面板合成与 HUD
- [vis/data_vis/viewer/viewer.py](../vis/data_vis/viewer/viewer.py) — OpenCV 交互窗口：帧滑条 + 键盘播放/单步/图层切换/截图

### vis/pred_vis/ — 感知模型预测可视化（加载权重，渲染三头预测与 GT 对照）

- [vis/pred_vis/__init__.py](../vis/pred_vis/__init__.py) — 感知模型预测可视化子模块包标识：加载权重、渲染双头预测与 GT 对照
- [vis/pred_vis/render/render.py](../vis/pred_vis/render/render.py) — 渲染：把感知模型双头预测（及可选 GT）着色并合成多帧多模态对照画布
- [vis/pred_vis/run.py](../vis/pred_vis/run.py) — 预测可视化入口 CLI：加载配置与权重 → 对场景逐帧推理 → 渲染预测与 GT 对照并保存

### vis/driving_vis/ — 驾驶模型可视化（透视模态 + GT/预测 三场/道路线/交通控制/轨迹）

- [vis/driving_vis/__init__.py](../vis/driving_vis/__init__.py) — 驾驶模型可视化子模块：对照渲染透视模态与 GT/预测三场、道路线、交通控制及轨迹。
- [vis/driving_vis/render/__init__.py](../vis/driving_vis/render/__init__.py) — 驾驶模型可视化渲染：三场/道路线/交通控制/多模态轨迹着色与混合尺寸面板合成。公开 API 重导出入口。
- [vis/driving_vis/render/render.py](../vis/driving_vis/render/render.py) — 渲染：把驾驶模型三场、道路线、交通控制与多模态轨迹着色，并和透视模态合成对照画布。
- [vis/driving_vis/run.py](../vis/driving_vis/run.py) — 驾驶可视化入口 CLI：逐帧渲染透视模态与 GT/预测三场、道路线、交通控制及轨迹并保存。
