"""闭环 RGB、碰撞与压线传感器的公开 API 重导出入口。

模块: clone_loop/worker/sensors/__init__.py
依赖: clone_loop.worker.sensors.sensors
读取配置: —
对外接口:
    - ClosedLoopSensors(world, ego, cfg_camera, fov_deg)
"""

from clone_loop.worker.sensors.sensors import ClosedLoopSensors

__all__ = ["ClosedLoopSensors"]
