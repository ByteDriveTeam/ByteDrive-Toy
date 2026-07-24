"""在 CARLA 全局路线中跟踪主车进度，并生成模型所需的 ego 系近端目标。

模块: clone_loop/worker/navigation/navigation.py
依赖: math, numpy, carla, agents.navigation.global_route_planner,
      clone_loop.worker.navigation.checks.navigation_checks
读取配置:
    clone_loop.route.sampling_resolution_m / target_distance_m / completion_distance_m /
        progress_search_points（由 cfg_route 传入）
对外接口:
    - RouteNavigator(world_map, start, end, cfg_route)
        .observe(ego_transform) -> dict
说明: 全局规划只在 episode 开始执行一次；每步仅在当前进度前方的有限窗口内向量化找最近点，避免回跳。
"""

import math

import carla
import numpy as np
from agents.navigation.global_route_planner import GlobalRoutePlanner

from clone_loop.worker.navigation.checks.navigation_checks import check_route_trace


__all__ = ["RouteNavigator"]


class RouteNavigator:
    """全局路线缓存与单调进度跟踪器。"""

    def __init__(self, world_map, start, end, cfg_route):
        self._cfg = cfg_route
        self._end = carla.Location(x=end[0], y=end[1], z=end[2])
        origin = carla.Location(x=start[0], y=start[1], z=start[2])
        trace = GlobalRoutePlanner(
            world_map, cfg_route.sampling_resolution_m).trace_route(origin, self._end)
        check_route_trace(trace)
        self._points = np.asarray([
            [item[0].transform.location.x, item[0].transform.location.y,
             item[0].transform.location.z] for item in trace
        ], dtype=np.float64)
        segments = np.linalg.norm(np.diff(self._points[:, :2], axis=0), axis=1)
        self._arc = np.r_[0.0, np.cumsum(segments)]
        self._progress = 0

    def observe(self, ego_transform):
        """更新单调路线进度，返回 ego 系目标、偏离量、完成度和终点距离。"""
        location = ego_transform.location
        current = np.array([location.x, location.y], dtype=np.float64)
        end_search = min(
            len(self._points), self._progress + self._cfg.progress_search_points)
        candidates = self._points[self._progress:end_search, :2]
        nearest_offset = int(np.argmin(np.linalg.norm(candidates - current[None], axis=1)))
        self._progress += nearest_offset
        deviation = float(np.linalg.norm(self._points[self._progress, :2] - current))
        target_arc = self._arc[self._progress] + self._cfg.target_distance_m
        target_index = min(int(np.searchsorted(self._arc, target_arc)), len(self._points) - 1)
        target_world = self._points[target_index, :2]
        target_local = _world_xy_to_ego(target_world, current, ego_transform.rotation.yaw)
        end_distance = location.distance(self._end)
        completion = float(self._arc[self._progress] / max(self._arc[-1], np.finfo(float).eps))
        return {
            "target_point": target_local.tolist(),
            "route_deviation_m": deviation,
            "route_completion": completion,
            "end_distance_m": float(end_distance),
            "reached": bool(end_distance <= self._cfg.completion_distance_m),
        }


def _world_xy_to_ego(point, origin, yaw_deg):
    """把世界平面点旋转平移到 CARLA 左手 ego 系（x 前、y 右）。"""
    delta = point - origin
    yaw = math.radians(yaw_deg)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return np.array([
        cosine * delta[0] + sine * delta[1],
        -sine * delta[0] + cosine * delta[1],
    ], dtype=np.float64)
