"""CARLA 原生交通灯控制关系与 Agent 规划关联。公开 API 重导出入口。

模块: worker/traffic_control/__init__.py
依赖: worker.traffic_control.traffic_control
读取配置: —
对外接口:
    - traffic_light_metadata(traffic_lights) -> list[dict]
    - traffic_light_states(traffic_lights) -> list[dict]
    - relevant_traffic_control(ego, agent, metadata, states) -> dict
说明: 跨模块统一 `from worker.traffic_control import ...`；实现见 traffic_control.py。
"""

from worker.traffic_control.traffic_control import (
    relevant_traffic_control,
    traffic_light_metadata,
    traffic_light_states,
)

__all__ = [
    "traffic_light_metadata",
    "traffic_light_states",
    "relevant_traffic_control",
]
