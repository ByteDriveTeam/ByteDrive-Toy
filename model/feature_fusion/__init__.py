"""DINO 多层完整序列融合：逐层 RMSNorm 后拼接并线性降到预测主干工作维。公开 API 重导出入口。

模块: model/feature_fusion/__init__.py
依赖: model.feature_fusion.feature_fusion
读取配置: —（实现文件经参数接收，本文件不读 config）
对外接口:
    - DinoFeatureFusion(hidden_dim, num_layers, out_channels) -> nn.Module   # DINO 多层特征融合
说明: 跨模块统一 `from model.feature_fusion import ...`；实现见 feature_fusion.py，入参校验见 checks/。
"""

from model.feature_fusion.feature_fusion import DinoFeatureFusion

__all__ = ["DinoFeatureFusion"]
