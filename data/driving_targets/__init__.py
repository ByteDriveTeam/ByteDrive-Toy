"""驾驶监督目标编码（numpy/OpenCV）：BEV/轨迹/三场、可见运动占用及八类多标签行为。公开 API 重导出入口。

模块: data/driving_targets/__init__.py
依赖: data.driving_targets.driving_targets
读取配置: —
对外接口:
    - BEHAVIOR_CLASSES / BehaviorParams / BevParams
    - bev_cell_centers / ego_xy_to_pixel / inview_mask / speed_accelerations
    - trajectory_targets / behavior_targets / risk_field / visible_moving_box_occupancy / distribution_field
说明: 跨模块统一 `from data.driving_targets import ...`；实现见 driving_targets.py，校验见 checks/。
"""

from data.driving_targets.driving_targets import (
    BEHAVIOR_CLASSES,
    BehaviorParams,
    BevParams,
    behavior_targets,
    bev_cell_centers,
    distribution_field,
    ego_xy_to_pixel,
    inview_mask,
    risk_field,
    speed_accelerations,
    trajectory_targets,
    visible_moving_box_occupancy,
)

__all__ = [
    "BEHAVIOR_CLASSES",
    "BehaviorParams",
    "BevParams",
    "bev_cell_centers",
    "ego_xy_to_pixel",
    "inview_mask",
    "speed_accelerations",
    "trajectory_targets",
    "behavior_targets",
    "risk_field",
    "visible_moving_box_occupancy",
    "distribution_field",
]
