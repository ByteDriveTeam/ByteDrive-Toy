"""Carla 世界/地图生命周期：连接、加载 Opt 地图、严格同步、天气与种子。公开 API 重导出入口。

模块: worker/session/__init__.py
依赖: worker.session.session
读取配置: —（参数由调用方经 cfg 传入）
对外接口:
    - connect / query_spawn_points               # 连接与可达点查询
    - load_scene_world                           # 加载地图并置严格同步
    - list_weather_presets / apply_weather       # 天气预设枚举与应用
说明: 跨模块统一 `from worker import session`（或 `from worker.session import ...`）；实现见 session.py，入参校验见 checks/。
"""

from worker.session.session import (
    apply_weather,
    connect,
    list_weather_presets,
    load_scene_world,
    query_spawn_points,
)

__all__ = [
    "connect", "query_spawn_points", "load_scene_world",
    "list_weather_presets", "apply_weather",
]
