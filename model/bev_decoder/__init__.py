"""统一 BEV 解码头：共享一次上采样，同时输出三场、道路线与交通控制预测。公开 API 重导出入口。

模块: model/bev_decoder/__init__.py
依赖: model.bev_decoder.bev_decoder
读取配置: —
对外接口:
    - BevDecoder(cfg_driving) -> nn.Module   # 统一空间驾驶任务解码头
"""

from model.bev_decoder.bev_decoder import BevDecoder

__all__ = ["BevDecoder"]
