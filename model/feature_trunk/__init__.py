"""预测特征主干：完整继承 DINOv3 Token 序列并经三层图像 Pre-Norm Transformer。公开 API 重导出入口。

模块: model/feature_trunk/__init__.py
依赖: model.feature_trunk.feature_trunk
读取配置: —
对外接口:
    - FeatureTrunk(...) -> nn.Module   # 完整图像 Token 序列的三层 Pre-Norm Transformer
说明: 跨模块统一 `from model.feature_trunk import ...`；实现见 feature_trunk.py，入参校验见 checks/。
"""

from model.feature_trunk.feature_trunk import FeatureTrunk

__all__ = ["FeatureTrunk"]
