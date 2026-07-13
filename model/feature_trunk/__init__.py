"""特征主干：已融合到工作维的单帧特征，经多层 2D 瓶颈残差块提炼空间表征。公开 API 重导出入口。

模块: model/feature_trunk/__init__.py
依赖: model.feature_trunk.feature_trunk
读取配置: —
对外接口:
    - FeatureTrunk(...) -> nn.Module   # 2D 单帧特征主干
说明: 跨模块统一 `from model.feature_trunk import ...`；实现见 feature_trunk.py，入参校验见 checks/。
"""

from model.feature_trunk.feature_trunk import FeatureTrunk

__all__ = ["FeatureTrunk"]
