"""BEV 编码器：融合图像与历史并提供六层主干末端及第 3/6 层规划特征。公开 API 重导出入口。

模块: model/bev_encoder/__init__.py
依赖: model.bev_encoder.bev_encoder
读取配置: —（转由 BevEncoder 读取 config.model.driving 相关键）
对外接口:
    - BevEncoder(cfg_driving) -> nn.Module
说明: 跨模块统一 `from model.bev_encoder import BevEncoder`；实现见 bev_encoder.py，校验见 checks/。
"""

from model.bev_encoder.bev_encoder import BevEncoder

__all__ = ["BevEncoder"]
