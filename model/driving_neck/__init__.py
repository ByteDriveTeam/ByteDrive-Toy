"""驾驶前端 neck：trunk+DINO 融合 + frustum 几何编码 + 残差。公开 API 重导出入口。

模块: model/driving_neck/__init__.py
依赖: model.driving_neck.driving_neck
读取配置: —（转由 DrivingNeck 读取 config.model.driving 相关键）
对外接口:
    - DrivingNeck(cfg_driving, trunk_channels, dino_channels, patch_size) -> nn.Module
说明: 跨模块统一 `from model.driving_neck import DrivingNeck`；实现见 driving_neck.py，校验见 checks/。
"""

from model.driving_neck.driving_neck import DrivingNeck

__all__ = ["DrivingNeck"]
