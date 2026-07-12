"""carla 几何对象与纯数值的转换，及相机内参推导。公开 API 重导出入口。

模块: worker/geometry/__init__.py
依赖: worker.geometry.geometry
读取配置: —
对外接口:
    - transform_to_list / location_to_list / make_transform   # carla<->list 位姿转换
    - bbox_to_dict                                            # 包围框→dict
    - compute_intrinsics(width, height, fov_deg)              # 相机内参推导
说明: 跨模块统一 `from worker.geometry import ...`；实现见 geometry.py（无校验）。
"""

from worker.geometry.geometry import (
    bbox_to_dict,
    compute_intrinsics,
    location_to_list,
    make_transform,
    transform_to_list,
)

__all__ = [
    "transform_to_list", "location_to_list", "make_transform",
    "bbox_to_dict", "compute_intrinsics",
]
