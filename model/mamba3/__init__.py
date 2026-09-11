"""单层纯 PyTorch Mamba-3 的公开 API 重导出入口。

模块: model/mamba3/__init__.py
依赖: model.mamba3.mamba3
读取配置: —（全部参数由第三方调用方传入）
对外接口:
    - Mamba3(...) -> nn.Module
    - Mamba3State(angle, ssm, prev_b, prev_x) -> state
    - mamba3_forward(x, layer, mask=None, state=None, return_state=False)
        -> Tensor 或 (Tensor, Mamba3State)
说明: 只提供单层算子；堆叠、残差和任务头由外部模型决定。
"""

from model.mamba3.mamba3 import Mamba3, Mamba3State, mamba3_forward

__all__ = ["Mamba3", "Mamba3State", "mamba3_forward"]
