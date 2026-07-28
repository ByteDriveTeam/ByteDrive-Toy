"""BEV 专用像素洗牌上采样器：以空间卷积和激活残差逐级恢复高分辨率特征。公开 API 重导出入口。

模块: model/bev_upsampler/__init__.py
依赖: model.bev_upsampler.bev_upsampler
读取配置: —
对外接口:
    - BevUpsampler(...) -> nn.Module   # 驾驶 BEV 分支专用级联上采样器
"""

from model.bev_upsampler.bev_upsampler import BevUpsampler

__all__ = ["BevUpsampler"]
