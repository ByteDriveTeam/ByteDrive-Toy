"""传感器阵列：逐视角按开关创建 RGB/Depth/语义/光流相机、语义分割 Lidar、碰撞传感器。公开 API 重导出入口。

模块: worker/sensors/__init__.py
依赖: worker.sensors.sensors
读取配置: —（参数由调用方经 cfg 传入）
对外接口:
    - SensorRig(...)   # 传感器阵列生命周期管理
说明: 跨模块统一 `from worker.sensors import ...`；实现见 sensors.py，入参校验见 checks/。
"""

from worker.sensors.sensors import SensorRig

__all__ = ["SensorRig"]
