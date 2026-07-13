"""驾驶监督目标编码（纯 numpy 函数）：BEV 几何/视场/轨迹/扇区/风险场/分布场。公开 API 重导出入口。

模块: data/driving_targets/__init__.py
依赖: data.driving_targets.driving_targets
读取配置: —
对外接口:
    - BevParams / bev_cell_centers / ego_xy_to_pixel / inview_mask
    - trajectory_targets / sector_of / risk_field / distribution_field
说明: 跨模块统一 `from data.driving_targets import ...`；实现见 driving_targets.py，校验见 checks/。
"""

from data.driving_targets.driving_targets import (
    BevParams,
    bev_cell_centers,
    distribution_field,
    ego_xy_to_pixel,
    inview_mask,
    risk_field,
    sector_of,
    trajectory_targets,
)

__all__ = [
    "BevParams",
    "bev_cell_centers",
    "ego_xy_to_pixel",
    "inview_mask",
    "trajectory_targets",
    "sector_of",
    "risk_field",
    "distribution_field",
]
