"""单场景异频采集循环：分别产出传感器帧与运动学时间轴。公开 API 重导出入口。

模块: worker/collect/__init__.py
依赖: worker.collect.collect
读取配置: —（参数由调用方经 cfg 传入）
对外接口:
    - prepare_drive(...)   # 规划路线并预热世界
    - collect_chunk(...)   # 采集一段直到 arena 写满/结束，返回两条时间轴
说明: 跨模块统一 `from worker import collect`（或 `from worker.collect import ...`）；实现见 collect.py，入参校验见 checks/。
"""

from worker.collect.collect import collect_chunk, prepare_drive

__all__ = ["prepare_drive", "collect_chunk"]
