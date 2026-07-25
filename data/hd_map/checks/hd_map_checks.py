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


def check_traffic_control_inputs(route_xy, state_names, relevant_control=None):
    """校验对象: HdMap.traffic_control_bev —— 路线、灯色类别及可选原生控制标注须满足契约。"""
    route = np.asarray(route_xy)
    if route.ndim != 2 or route.shape[1] != 2:
        raise ValueError("route_xy 期望 [N,2]，实际 {}。".format(tuple(route.shape)))
    if not state_names or "red" not in state_names:
        raise ValueError("state_names 须非空且包含 red。")
    if relevant_control is None:
        return
    if not isinstance(relevant_control, dict) or not isinstance(relevant_control.get("valid"), bool):
        raise ValueError("relevant_control 须为含布尔 valid 的 dict。")
    if not relevant_control["valid"]:
        return
    required = {
        "stop_location", "stop_yaw", "lane_width", "route_distance", "state",
    }
    missing = sorted(required - set(relevant_control))
    if missing:
        raise KeyError("有效 relevant_control 缺字段: {}。".format(missing))
    if np.asarray(relevant_control["stop_location"]).shape != (3,):
        raise ValueError("relevant_control.stop_location 期望 [3]。")
    if relevant_control["lane_width"] <= 0 or relevant_control["route_distance"] < 0:
        raise ValueError("relevant_control 车道宽度须为正、路线距离须非负。")
