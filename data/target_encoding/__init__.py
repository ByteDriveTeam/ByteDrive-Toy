"""监督目标编码：Symlog 物理量、深度范围掩码的纯函数。公开 API 重导出入口。

模块: data/target_encoding/__init__.py
依赖: data.target_encoding.target_encoding
读取配置: —（纯函数，参数由调用方传入）
对外接口:
    - symlog / inv_symlog           # Symlog 正/逆变换
    - physics_target / physics_decode   # 物理量编码/解码
    - depth_targets                 # 深度目标与范围掩码
说明: 跨模块统一 `from data.target_encoding import ...`（或 `from data import target_encoding as te`）；
      实现见 target_encoding.py，入参校验见 checks/。
"""

from data.target_encoding.target_encoding import (
    depth_targets,
    inv_symlog,
    physics_decode,
    physics_target,
    symlog,
)

__all__ = [
    "symlog", "inv_symlog", "physics_target", "physics_decode",
    "depth_targets",
]
