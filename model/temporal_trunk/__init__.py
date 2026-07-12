"""时序主干：DINO 逐帧特征先 1×1×1 投影降维，堆成时序后经多层 3D ConvNeXt 块提炼时空表征。公开 API 重导出入口。

模块: model/temporal_trunk/__init__.py
依赖: model.temporal_trunk.temporal_trunk
读取配置: —
对外接口:
    - TemporalTrunk(...) -> nn.Module   # 3D 时序主干
说明: 跨模块统一 `from model.temporal_trunk import ...`；实现见 temporal_trunk.py，入参校验见 checks/。
"""

from model.temporal_trunk.temporal_trunk import TemporalTrunk

__all__ = ["TemporalTrunk"]
