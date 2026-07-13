"""三场解码头：像素洗牌上采样 → 风险/可行驶/轨迹分布场。公开 API 重导出入口。

模块: model/field_decoder/__init__.py
依赖: model.field_decoder.field_decoder
读取配置: —（转由 FieldDecoder 读取 config.model.driving 相关键）
对外接口:
    - FieldDecoder(cfg_driving) -> nn.Module
说明: 跨模块统一 `from model.field_decoder import FieldDecoder`；实现见 field_decoder.py，校验见 checks/。
"""

from model.field_decoder.field_decoder import FieldDecoder

__all__ = ["FieldDecoder"]
