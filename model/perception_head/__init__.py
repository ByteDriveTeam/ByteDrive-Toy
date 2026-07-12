"""感知解码头：3D 残差块 + 通道压缩 + 级联像素洗牌上采样至原分辨率。公开 API 重导出入口。

模块: model/perception_head/__init__.py
依赖: model.perception_head.perception_head
读取配置: —
对外接口:
    - PerceptionHead(...) -> nn.Module   # 单任务感知解码头
说明: 跨模块统一 `from model.perception_head import ...`；实现见 perception_head.py，入参校验见 checks/。
"""

from model.perception_head.perception_head import PerceptionHead

__all__ = ["PerceptionHead"]
