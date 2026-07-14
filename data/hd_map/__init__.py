"""HD 地图：车道折线 → ego BEV 可行驶掩码与越界距离场。公开 API 重导出入口。

模块: data/hd_map/__init__.py
依赖: data.hd_map.hd_map
读取配置: —
对外接口:
    - HdMap(npz_path) -> HdMap   # .drivable_bev(ego_pose6, bev, lane_half_width_m)
    - offroad_distance_field(drivable, bev) -> ndarray
说明: 跨模块统一 `from data.hd_map import HdMap`；实现见 hd_map.py，校验见 checks/。
"""

from data.hd_map.hd_map import HdMap, offroad_distance_field

__all__ = ["HdMap", "offroad_distance_field"]
