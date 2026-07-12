"""纯 numpy 复刻 CARLA 坐标变换与 3D->2D 投影。公开 API 重导出入口。

模块: vis/data_vis/geometry/__init__.py
依赖: vis.data_vis.geometry.geometry
读取配置: —
对外接口:
    - transform_matrix / intrinsic_matrix        # 位姿→4x4、内参→3x3
    - world_to_camera / world_to_ego             # 世界系到相机/ego 系变换
    - transform_points / bbox_corners            # 点变换、包围框角点
    - project_points                             # 3D→2D 投影
说明: 跨模块统一 `from vis.data_vis import geometry`（或 `from vis.data_vis.geometry import ...`）；
      实现见 geometry.py，入参校验见 checks/。
"""

from vis.data_vis.geometry.geometry import (
    bbox_corners,
    intrinsic_matrix,
    project_points,
    transform_matrix,
    transform_points,
    world_to_camera,
    world_to_ego,
)

__all__ = [
    "transform_matrix", "intrinsic_matrix", "world_to_camera", "world_to_ego",
    "transform_points", "bbox_corners", "project_points",
]
