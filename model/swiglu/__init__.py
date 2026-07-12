"""通用 SwiGLU 激活模块（沿维度二等分为 value/gate）：公开 API 重导出入口。

模块: model/swiglu/__init__.py
依赖: model.swiglu.swiglu
读取配置: —
对外接口:
    - swiglu(features, dim=-1) -> Tensor   # 沿 dim 二等分为 value/gate 后输出 value * silu(gate)
    - SwiGLU(dim=-1) -> nn.Module          # 上述激活的层封装
说明: 跨模块统一 `from model.swiglu import ...`；实现见 swiglu.py，入参校验见 checks/。
"""

from model.swiglu.swiglu import SwiGLU, swiglu

__all__ = ["SwiGLU", "swiglu"]
