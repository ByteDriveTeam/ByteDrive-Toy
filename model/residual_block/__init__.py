"""视觉编码器残差卷积模块（1D/2D/3D RMSNorm、瓶颈残差块与 3D ConvNeXt 块）：公开 API 重导出入口。

模块: model/residual_block/__init__.py
依赖: model.residual_block.residual_block
读取配置: —
对外接口:
    - RMSNorm1d / RMSNorm2d / RMSNorm3d -> nn.Module   # 各维度 RMSNorm
    - ResidualBlock1d / ResidualBlock / ResidualBlock3d -> nn.Module   # 瓶颈残差块
    - ConvNeXtBlock3d -> nn.Module                       # 3D ConvNeXt 块
说明: 跨模块统一 `from model.residual_block import ...`；实现见 residual_block.py，入参校验见 checks/。
"""

from model.residual_block.residual_block import (
    ConvNeXtBlock3d,
    RMSNorm1d,
    RMSNorm2d,
    RMSNorm3d,
    ResidualBlock,
    ResidualBlock1d,
    ResidualBlock3d,
)

__all__ = [
    "RMSNorm1d",
    "RMSNorm2d",
    "RMSNorm3d",
    "ResidualBlock1d",
    "ResidualBlock",
    "ResidualBlock3d",
    "ConvNeXtBlock3d",
]
