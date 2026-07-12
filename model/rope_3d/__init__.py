"""通用 3D RoPE 旋转位置编码（只消费调用方传入的三维坐标，全程 FP32）：公开 API 重导出入口。

模块: model/rope_3d/__init__.py
依赖: model.rope_3d.rope_3d
读取配置: —
对外接口:
    - apply_rope_3d(...) -> Tensor   # 对输入按三维坐标施加 3D RoPE
    - RoPE3D(...) -> nn.Module        # 3D RoPE 的模块封装
说明: 跨模块统一 `from model.rope_3d import ...`；实现见 rope_3d.py，入参校验见 checks/。
"""

from model.rope_3d.rope_3d import RoPE3D, apply_rope_3d

__all__ = ["RoPE3D", "apply_rope_3d"]
