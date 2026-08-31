# BEV 世界模型与联合训练

## 数据契约

离线生成器消费 `data/carla_data_collector/dataset/scenes/scene_*/lmdb`，输出
`data/bev_grid_generation/output/scene_*/lmdb`。单帧 shape 为 `(10, 256, 256)`，
BEV 前/后/左/右各 32 m，分辨率 0.25 m，自车位于中心。图层顺序固定为：

1. 红灯停止线
2. 黄灯停止线
3. 绿灯停止线
4. 车辆
5. 行人
6. 可行驶区域
7. 车道中心线
8. 车道分隔线
9. 道路边界
10. 其他车道线

交通灯状态绑定到该灯的全部 CARLA 原生 `stop_waypoints`；车辆、行人和地图信息不按
自车路线或相关性筛选。每帧二值数组使用 little-endian `packbits` 后再 zlib 压缩，LMDB
键为 `grid/{frame:08d}`，元数据键为 `meta`。

## 模型

5 帧 10 Hz 栅格先经 `kernel_size=stride=16` 的卷积得到每帧 `16×16` Token。Student
在单帧空间组合中采样 75% 掩码并向 5 帧广播，物理删除掩码 Token；Teacher 接收全部 Token。

Encoder 为 12 个 Pre-Norm Transformer Block，特征维 512、16 头。每个 Block 的 SDPA
和 FFN 都视为一个独立子层；第 `k` 个子层对全部历史输出学习 logits，Softmax 后得到当前输入：

`x_k = sum_i softmax(logits_k)[i] * history_i`

每列 logits 均全零初始化，所以初始 Softmax 对该层可见的所有历史输出严格等权相加，避免人为偏向某个深度；之后再执行该子层的普通 Pre-Norm 残差。注意力使用逐头 QKNorm、3D RoPE `(row,column,time)`
和 PyTorch SDPA；FFN 为 `Linear(D,4D) -> SwiGLU(4D,2D) -> Linear(2D,D)`。

Predictor 取 Student 第 3/6/9/12 个完整 Block 输出，分别 RMSNorm 后在通道维拼接并降到
256 维，再补可学习 MaskToken。六层、8 头 Predictor 输出升到 2048，拆成四个 512 维目标并
分别 RMSNorm，与 EMA Teacher 对应层的 RMSNorm 输出做均方误差。Teacher 更新为
`teacher = (1 - 0.00075) * teacher + 0.00075 * student`。

## 损失与训练

掩码位置权重恒为 1。可见位置按到最近空间掩码的米制距离和距最新帧的时间距离指数衰减，
并在配置的优化步数内从 0 线性升温。

训练不再划分阶段。每个优化步同时执行掩码补全和 VISReg：重建目标驱动 Predictor 对齐 EMA
Teacher 的滞后状态，并向 Student 提供掩码不变性约束；VISReg 直接作用于 Student 骨干末端输出
对全部可见时空 Token 做 GAP 后的 `[B,D]` 特征，用 center、scale 和 sliced-Wasserstein shape
三项约束特征分布，不再生成第二掩码视图或计算 invariance 项。默认单阶段训练 50 epoch，保持
原三阶段合计的训练预算。

为让 VISReg 的 batch 统计覆盖梯度累计后的有效 batch，训练循环采用两遍梯度缓存。第一遍固定
每个样本的重建掩码，在无梯度模式下逐微批收集 GAP，拼成有效 batch 后计算 VISReg 对 GAP 的
梯度；第二遍以相同样本和掩码逐微批重放，在一次反传中同时加入 Predictor/Teacher 重建损失和
缓存的 GAP 梯度。因此 `batch_size=1` 时可通过 `grad_accum_steps>=2` 获得有效的 VISReg 统计，
且无需跨微批保留 Encoder 计算图。若数据尾部只剩一个样本，该优化步仍执行重建并明确记录
`visreg_skipped_small_batch=1`。每个优化步在裁剪前监控全局/逐参数梯度范数和 NaN/Inf。

## PowerShell 命令

```powershell
# 先用一个场景验证离线生成
.\.venv\Scripts\python.exe -m data.bev_grid_generation.run --scene-limit 1

# 保存一帧可视化；加 --show 可播放
.\.venv\Scripts\python.exe -m vis.bev_grid_vis.run --scene scene_000000 --frame 0

# 按配置执行单阶段联合训练
.\.venv\Scripts\python.exe -m train.run_world_model
```

生成器对带 `complete=true` 的目标场景自动跳过，支持中断后按场景续跑。`torch.compile` 默认关闭；
仅在 CUDA 且显式启用时尝试编译 Student/Predictor；Windows 会启用 Dynamo 错误抑制，
初始化或首次真实图编译失败时回退 eager。
