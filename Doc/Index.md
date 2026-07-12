# ByteDrive 文档索引

全项目文档与源文件的单一导航入口。**每次增删文件，同一提交内更新本表。**
每行格式：`相对路径 — 一句话职责`（职责应与该文件文件头首行一致）。
校验伴随文件 `X_checks.py` 附属于 `X.py`，**不单独列入索引**。

## 规范与文档

- [Doc/开发规范.md](开发规范.md) — 项目强制开发规范（文档/注释/配置/校验/简洁）
- [Doc/Index.md](Index.md) — 本文档索引

## config/ — 配置与校验（参数唯一来源）

- [config/default.yaml](../config/default.yaml) — ByteDrive 全部参数默认值（唯一数据来源）
- [config/schema.py](../config/schema.py) — 配置的类型定义与加载期校验（参数约束的唯一来源）
- [config/__init__.py](../config/__init__.py) — 配置加载入口：读 yaml → 构造 schema → 校验 → 返回配置对象

## data/ — 数据读取与预处理

- [data/__init__.py](../data/__init__.py) — 数据读取与预处理包标识：只读消费 config 与已落盘数据集
- [data/target_encoding.py](../data/target_encoding.py) — 监督目标编码：Symlog 物理量、深度范围掩码、光流→图像平面速度的纯函数
- [data/perception_dataset.py](../data/perception_dataset.py) — 感知模型时序开窗数据集：把落盘场景切成 5 帧窗口，产出归一化 RGB 与四任务监督目标

### data/carla_data_collector/ — Carla 合成数据采集（Py37 worker + Py312 collector 异构）

- [data/carla_data_collector/README.md](../data/carla_data_collector/README.md) — 本采集模块的设计文档（架构/数据流/输出布局/运行）
- [scene_layout.py](../data/carla_data_collector/scene_layout.py) — 从 Carla 地图提取静态/动态场景布局（官方示例改造）

共享层 `common/`（两端纯 Python，3.7/3.12 双兼容）

- [common/protocol.py](../data/carla_data_collector/common/protocol.py) — 控制管道 JSON 行命令/响应协议与帧索引/语义Lidar dtype 定义
- [common/shm.py](../data/carla_data_collector/common/shm.py) — 匿名共享内存 arena：跨进程零拷贝传大块数据，兼作场景内存缓冲

Py37 采集端 `worker/`（仅 py37_venv 运行）

- [worker/main.py](../data/carla_data_collector/worker/main.py) — Py37 worker 子进程入口：经 stdin/stdout JSON 协议受 collector 驱动采集
- [worker/session.py](../data/carla_data_collector/worker/session.py) — Carla 世界/地图生命周期：连接、加载 Opt 地图、严格同步、天气与种子
- [worker/actors.py](../data/carla_data_collector/worker/actors.py) — 主车、交通流与行人的生成与销毁
- [worker/sensors.py](../data/carla_data_collector/worker/sensors.py) — 传感器阵列：逐视角按开关创建 RGB/Depth/语义/光流相机、语义分割 Lidar、碰撞传感器
- [worker/annotations.py](../data/carla_data_collector/worker/annotations.py) — 带语义的包围框抽取：动态 actor（逐帧）与静态环境物体（每场景）
- [worker/collect.py](../data/carla_data_collector/worker/collect.py) — 单场景严格同步采集循环：逐帧收齐传感器、交通灯状态、共享内存数据与帧索引
- [worker/geometry.py](../data/carla_data_collector/worker/geometry.py) — carla 几何对象与纯数值的转换，及相机内参推导

Py312 编排处理端 `collector/`（根 .venv 运行）

- [collector/worker_proc.py](../data/carla_data_collector/collector/worker_proc.py) — 派生并驱动 Py37 worker 子进程的控制管道客户端
- [collector/routes.py](../data/carla_data_collector/collector/routes.py) — 由可达点构建路线队列：两两组合、按直线距离过滤、随机排序、确保不重复
- [collector/scenarios.py](../data/carla_data_collector/collector/scenarios.py) — 逐场景随机：种子与天气预设（决策与记录都在 collector 侧，便于复现）
- [collector/encode.py](../data/carla_data_collector/collector/encode.py) — 把单相机的 BGR 帧序列编码为 H.265 mp4
- [collector/writer.py](../data/carla_data_collector/collector/writer.py) — 把场景的非 RGB 数据写入 LMDB（深度/语义Lidar/包围框/主车状态/元数据/视频引用）
- [collector/orchestrator.py](../data/carla_data_collector/collector/orchestrator.py) — 采集主循环：建队列→驱动 worker→碰撞重试→读共享内存→编码+写 LMDB
- [collector/run.py](../data/carla_data_collector/collector/run.py) — Py312 采集入口 CLI：加载配置并启动采集主循环

## model/ — 网络结构定义

- [model/__init__.py](../model/__init__.py) — 网络结构定义包标识：只读消费 config，不含可调参数默认值
- [model/swiglu.py](../model/swiglu.py) — 通用 SwiGLU 激活模块（沿维度二等分为 value/gate）
- [model/rope_3d.py](../model/rope_3d.py) — 通用 3D RoPE 旋转位置编码（只消费调用方传入的三维坐标，全程 FP32）
- [model/residual_block.py](../model/residual_block.py) — 视觉编码器残差卷积模块（1D/2D/3D RMSNorm、瓶颈残差块与 3D ConvNeXt 块）
- [model/dinov3_backbone.py](../model/dinov3_backbone.py) — DINOv3 ViT-B 视觉骨干：全程冻结 + eval，逐帧输出 patch 网格特征
- [model/temporal_trunk.py](../model/temporal_trunk.py) — 时序主干：DINO 逐帧特征堆成时序后经多层 3D ConvNeXt 块提炼时空表征
- [model/pixel_shuffle_upsampler.py](../model/pixel_shuffle_upsampler.py) — 级联像素洗牌上采样：把低分辨率特征逐级 2× 放大回原分辨率
- [model/perception_head.py](../model/perception_head.py) — 感知解码头：3D 残差块 + 通道压缩 + 级联像素洗牌上采样至原分辨率
- [model/perception_model.py](../model/perception_model.py) — 多任务时序感知模型：冻结 DINOv3 骨干 + 3D 时序主干 + 语义/光流/深度三头
- [model/target_point_embedding.py](../model/target_point_embedding.py) — 目标点嵌入层：ego 目标点经栅格向量场与三层卷积编码为目标导航点 Token

## train/ — 训练 / 评估循环

- [train/__init__.py](../train/__init__.py) — 训练 / 优化 / 评估循环包标识：只读消费 config
- [train/losses.py](../train/losses.py) — 多任务监督损失：语义 CE + 深度 SmoothL1(掩码) + 深度范围 BCE + 光流 SmoothL1(掩码)
- [train/optimizer.py](../train/optimizer.py) — 优化器构造：仅优化可训练参数（主干+三头），骨干冻结不纳入
- [train/loop.py](../train/loop.py) — 训练与评估循环：前向 → 多任务损失 → 反向 → 梯度裁剪 → 步进，并聚合日志
- [train/run.py](../train/run.py) — 训练入口 CLI：加载配置 → 建模型/数据/优化器 → 逐 epoch 训练并保存权重

## clone_loop/ — 行为克隆闭环

- _待补充_

## vis/ — 可视化与日志渲染

- [vis/data_vis/__init__.py](../vis/data_vis/__init__.py) — 可视化包标识：只读消费采集数据集并渲染
- [vis/data_vis/run.py](../vis/data_vis/run.py) — 可视化入口 CLI：加载配置、定位场景目录、启动交互窗口
- [vis/data_vis/reader.py](../vis/data_vis/reader.py) — 场景读取器：合并单场景的 LMDB 与 mp4 为逐帧数据，探测各模态可用性
- [vis/data_vis/geometry.py](../vis/data_vis/geometry.py) — 纯 numpy 复刻 CARLA 坐标变换与 3D->2D 投影
- [vis/data_vis/palette.py](../vis/data_vis/palette.py) — CARLA 语义标签到颜色的调色板与向量化映射
- [vis/data_vis/draw.py](../vis/data_vis/draw.py) — 渲染：3D 框投影、深度/语义/光流着色、lidar+框 鸟瞰图、多面板合成与 HUD
- [vis/data_vis/viewer.py](../vis/data_vis/viewer.py) — OpenCV 交互窗口：帧滑条 + 键盘播放/单步/图层切换/截图

### vis/pred_vis/ — 感知模型预测可视化（加载权重，渲染三头预测与 GT 对照）

- [vis/pred_vis/__init__.py](../vis/pred_vis/__init__.py) — 感知模型预测可视化子模块包标识：加载权重、渲染三头预测与 GT 对照
- [vis/pred_vis/render.py](../vis/pred_vis/render.py) — 渲染：把感知模型三头预测（及可选 GT）着色并合成多帧多模态对照画布
- [vis/pred_vis/run.py](../vis/pred_vis/run.py) — 预测可视化入口 CLI：加载配置与权重 → 对场景逐窗推理 → 渲染预测与 GT 对照并保存
