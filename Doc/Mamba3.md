# Mamba-3 的精度、状态、分段扫描与训练接入约定

本模块提供单层纯 PyTorch 算子，结构参数仍由调用方显式传入，不在实现文件中引入新的模型默认配置。公开入口为 `from model.mamba3 import Mamba3, Mamba3State, mamba3_forward`；残差、堆叠和任务头由调用方组织。

## 精度

`dtype` 控制输入、输出线性投影参数；它们也支持 autocast。归一化、B/C bias、dt/A、相位、rank 归约、SSM 更新和状态读出使用 FP32，并在核心计算区域关闭 autocast。

除输入/输出投影外的可训练参数始终保持 FP32。调用 `.to(dtype=...)`、`.half()`、`.bfloat16()` 时，核心参数和已有梯度不会先被量化再恢复；设备移动仍生效。建议在构造优化器前完成设备和精度转换。参数键名保持不变。

`angle` 和 `ssm` 状态为 FP32。`allocate_state(batch_size, dtype=...)` 的 dtype 只控制 `prev_b/prev_x` 缓存；默认使用输入投影权重 dtype，forward 自动分配状态也遵循这一规则。旋转后的当前 B/C 使用 FP32 计算，写入 B 缓存时才按缓存 dtype 转换。

扫描和 step 的相位均按 `2π` 取模。group 按连续的 head 区间共享 B/C，例如四个 head、两个 group 对应 `[0,0,1,1]`。

RMSNorm 先按每组最大绝对值缩放，并同步缩放 epsilon，再计算均方，避免大幅度有限输入在平方时溢出。该变换保留归一化定义及梯度。A 的 heavy-tail 激活按互斥分支计算，零点导数为 1。

构造参数中的维度、rank 和 chunk_size 必须为正整数类型，拒绝浮点值与 bool 的隐式整数转换。稳定性常量必须为 FP32 可表示的有限正实数；模式开关必须为 bool。

## MIMO 初始化补偿

SISO 的 `D` 初值为 1，MIMO 的 `D` 初值为 `mimo_rank`。其余 MIMO 投影保持 `mimo_x=1/rank`、`mimo_z=1`、`mimo_o=1/rank`。在初始门控相同的条件下，补偿使 D 分支的有效直通增益不再随 rank 增大而缩小。

补偿同时用于启用和关闭输出归一化的 MIMO 模型；它只改变新建模型的 D 初值，不改参数键名、张量布局或已加载的 D 权重。它是基于 rank 推导的固定初始化规则，不新增调参常量。

## mask、reset 与空序列

`forward(x, mask=None, state=None, return_state=False, reset=None, *, checkpoint_chunks=False)` 接收 `[B,L,d_model]` 输入。`mask/reset` 均为同设备的 `[B,L]` bool 张量；step 使用 `[B,d_model]` 输入及 `[B]` 掩码。

- `mask=False`：输出为零，暂停相位、SSM 和 B/x 缓存更新；无效输入在投影前置零，因此允许 NaN/Inf 占位。
- `reset=True`：在对应 token **之前**清零该 batch 的全部状态，再决定是否处理 token；即使该 token 被 mask，也执行清零。
- 新序列边界使用 reset；单纯 mask 不会切断跨序列记忆。
- 空序列返回空输出，以及保留梯度连接的独立状态副本。

```python
output, state = layer(tokens, mask=valid, reset=sequence_starts, return_state=True)
output, state = layer.step(next_token, state, mask=next_valid, reset=next_sequence_start)
```

## 缓存与梯度

返回状态采用独立紧凑存储，避免最后一个 token 的切片保留整个 chunk。默认保留计算图，允许跨 forward/step 调用反向传播。

```python
state = state.detach()             # 显式截断反向传播，并复制存储
state = state.clone()              # 独立副本，保留梯度连接
state = state.reset(batch_starts)  # [B] bool，只清零指定 batch
```

推理时使用 `torch.no_grad()` 或 `torch.inference_mode()`；`detach()` 不会代替推理上下文。独立存储不等于截断计算图，训练中尚未反向的历史仍可由 autograd 保留。

## 内存与训练

rank 在张量收缩时归约，不再显式构造 `[B,L,R,H,P,N]` 外积。但参考扫描仍需生成 `[B,chunk,H,P,N]` 的完整状态序列，chunk 大小仍影响峰值。

```python
output, state = layer(
    tokens, mask=valid, reset=sequence_starts, state=state,
    return_state=True, checkpoint_chunks=True)
```

`checkpoint_chunks=True` 在反向时重计算每个 chunk 的激活，减少 autograd 保存量；保留跨 chunk、跨调用的状态梯度，支持 AMP 及输入不需要梯度的训练。它与普通 forward 使用相同公式、参数与返回布局。该开关选择本次调用的执行方式，不新增模型超参数；无梯度上下文直接执行普通计算。

普通与重计算调用均经过 `nn.Module.__call__`，因此执行本层的 forward pre/post hooks。兼容入口 `layer.forward_checkpointed(...)` 转发到 `layer(..., checkpoint_chunks=True)`，也保留本层 hooks。函数式 `mamba3_forward` 同样支持该关键字。使用 DDP/FSDP 等外部包装器时，应通过包装器正常调用并传递此关键字，避免直接调用 `.module` 上的方法而绕过外层包装器。

`fuse_pregate_headwise_norm` 保留为接口兼容参数，两种值在当前纯 PyTorch 实现中使用相同的按 head 归一化，不表示存在独立融合 kernel。

目前验证为 CPU 合成输入上的输出、状态、梯度及保存张量检查；不能据此认定 CUDA 峰值、吞吐或真实训练收益。大规模训练仍应在目标 GPU 上测量，再决定是否引入专用 kernel。

## 优化器

`dt_bias` 与 `D` 标记为 `_no_weight_decay`。项目 `train.optimizer.build_optimizer` 会在原有参数分组内拆分免衰减参数，保留学习率、betas 等组选项；普通 `nn.Module` 可直接接入。

外部直接构造 AdamW 时，应按该标记显式拆组；`torch.optim.AdamW(layer.parameters(), ...)` 不会自行识别自定义属性。优化器超参数继续来自调用方配置。

## 数值诊断

```python
metrics = layer.diagnostics(tokens, mask=valid, state=state)
```

诊断显式执行，不进入日常前向，也不修改参数或状态。结果包括：有效 token 数、dt/`|A·dt|`/相位增量绝对值的最小值、均值、95 分位数和最大值，`tanh` 精确饱和比例及平均导数，以及可选的 SSM RMS 和绝对值上界。无有效 token 时不生成分布指标。GPU 调用会因 `.item()` 同步，宜低频采样。

当前不额外截断 dt 或角速度；应先观察上述分布和训练损失，再决定是否通过配置引入软上限。相位取模只能限制累计相位，不能修复投影溢出、无限 dt 或已经产生的非有限状态。

## 验证

```powershell
.\.venv\Scripts\python.exe -B -m unittest model.mamba3.checks.test_mamba3 train.optimizer.checks.test_optimizer -v
```

独立串行参考不复用模型的投影、group、旋转或扫描辅助函数；覆盖多组合法 head/group 配置、mask/reset、非零初始状态、输出和梯度。另有零点梯度 gradcheck、RMSNorm 的 FP64 大数参考、构造参数拒绝、rank 直通增益、模块 hooks、FP16 溢出、autocast、状态独立存储、重计算、参数精度转换、免衰减与诊断检查。
