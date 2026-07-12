"""级联像素洗牌上采样：把低分辨率特征逐级 2× 放大回原分辨率。公开 API 重导出入口。

模块: model/pixel_shuffle_upsampler/__init__.py
依赖: model.pixel_shuffle_upsampler.pixel_shuffle_upsampler
读取配置: —
对外接口:
    - PixelShuffleUpsampler(...) -> nn.Module   # 级联像素洗牌上采样
说明: 跨模块统一 `from model.pixel_shuffle_upsampler import ...`；实现见 pixel_shuffle_upsampler.py，入参校验见 checks/。
"""

from model.pixel_shuffle_upsampler.pixel_shuffle_upsampler import PixelShuffleUpsampler

__all__ = ["PixelShuffleUpsampler"]
