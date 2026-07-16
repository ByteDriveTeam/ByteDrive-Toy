# 本文件为 data/hd_map/hd_map.py 的校验伴随文件（规范 §7.1，免文件头）。

from pathlib import Path

import numpy as np


def check_map_path(path):
    """校验对象: HdMap 构造入参 path —— HD 地图 npz 文件须存在。"""
    if not Path(path).is_file():
        raise FileNotFoundError("HD 地图文件不存在: {}（请将对应地图的 *_HD_map.npz 放入 data/map/）。".format(path))


def check_polylines(polylines, path):
    """校验对象: HdMap 解析结果 —— 至少解析出一条车道折线，否则地图为空/结构不符。"""
    if not polylines:
        raise ValueError("HD 地图 {} 未解析出任何车道折线（结构可能与预期不符）。".format(path))


def check_drivable_mask(drivable, bev):
    """校验对象: offroad_distance_field 入参 —— 可行驶掩码形状须与 BEV 分辨率一致。"""
    expected = (bev.height, bev.width)
    if tuple(drivable.shape) != expected:
        raise ValueError("drivable 期望形状 {}，实际 {}。".format(expected, tuple(drivable.shape)))


def check_traffic_control_inputs(route_xy, state_names):
    """校验对象: HdMap.traffic_control_bev —— 路线须为 [N,2]，灯色类别须非空且含 red。"""
    route = np.asarray(route_xy)
    if route.ndim != 2 or route.shape[1] != 2:
        raise ValueError("route_xy 期望 [N,2]，实际 {}。".format(tuple(route.shape)))
    if not state_names or "red" not in state_names:
        raise ValueError("state_names 须非空且包含 red。")
