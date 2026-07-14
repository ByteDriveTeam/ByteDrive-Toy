"""轨迹/行为联合解码器：8 扇区轨迹 Token 与行为 Token 组成同一序列。公开 API 重导出入口。

模块: model/trajectory_decoder/__init__.py
依赖: model.trajectory_decoder.trajectory_decoder
读取配置: —（转由 TrajectoryDecoder 读取 config.model.driving 相关键）
对外接口:
    - TrajectoryDecoder(cfg_driving) -> nn.Module
说明: 跨模块统一 `from model.trajectory_decoder import TrajectoryDecoder`；实现见 trajectory_decoder.py，校验见 checks/。
"""

from model.trajectory_decoder.trajectory_decoder import TrajectoryDecoder

__all__ = ["TrajectoryDecoder"]
