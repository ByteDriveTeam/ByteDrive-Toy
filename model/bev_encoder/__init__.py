"""BEV 编码器：融合三目图像 Token 与历史并提供多层规划特征的公开 API 入口。

模块: model/bev_encoder/__init__.py
依赖: model.bev_encoder.bev_encoder
读取配置: —（转由 BevEncoder 读取 config.model.driving 相关键）
对外接口:
    - BevEncoder(cfg_driving) -> nn.Module
说明: 跨模块统一 `from model.bev_encoder import BevEncoder`；实现见 bev_encoder.py，校验见 checks/。
"""

from model.bev_encoder.bev_encoder import BevEncoder

__all__ = ["BevEncoder"]
