"""主车、交通流与行人的生成与销毁。公开 API 重导出入口。

模块: worker/actors/__init__.py
依赖: worker.actors.actors
读取配置: —（参数由调用方经 cfg 传入）
对外接口:
    - spawn_ego / spawn_traffic_vehicles / spawn_walkers   # 生成主车/车流/行人
    - destroy_scene_actors                                 # 销毁本场景所有 actor
    - WalkerCrowd                                          # 行人群体管理
说明: 跨模块统一 `from worker import actors`（或 `from worker.actors import ...`）；实现见 actors.py，入参校验见 checks/。
"""

from worker.actors.actors import (
    WalkerCrowd,
    destroy_scene_actors,
    spawn_ego,
    spawn_traffic_vehicles,
    spawn_walkers,
)

__all__ = [
    "spawn_ego", "spawn_traffic_vehicles", "spawn_walkers",
    "destroy_scene_actors", "WalkerCrowd",
]
