"""独立道路线图解码器：输出类别 logits 与有向切向量。公开 API 重导出入口。

模块: model/lane_map_decoder/__init__.py
依赖: model.lane_map_decoder.lane_map_decoder
读取配置: —（转由 LaneMapDecoder 读取 config.model.driving 相关键）
对外接口:
    - LaneMapDecoder(cfg_driving) -> nn.Module
说明: 跨模块统一 `from model.lane_map_decoder import LaneMapDecoder`。
"""

from model.lane_map_decoder.lane_map_decoder import LaneMapDecoder

__all__ = ["LaneMapDecoder"]
