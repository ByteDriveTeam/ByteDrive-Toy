"""CARLA 原生交通灯车道拓扑与 Agent 当前规划关联，生成可落盘交通控制真值。

模块: worker/traffic_control/traffic_control.py
依赖: carla, numpy, worker.geometry
读取配置: —（关联完全采用 CARLA 车道/停止点真值与 Agent 当前规划，不设实验阈值）
对外接口:
    - traffic_light_metadata(traffic_lights) -> list[dict]  # 场景级灯、受控车道与停止点
    - traffic_light_states(traffic_lights) -> list[dict]    # 逐帧全部灯状态
    - relevant_traffic_control(ego, agent, metadata, states) -> dict  # 当前路线下一控制点
说明: actor id 只负责同一 episode 内逐帧状态关联，OpenDRIVE id 供跨 episode 稳定识别。CARLA 的
      affected_lane_waypoints 位于灯控路口内部，stop_waypoints 位于进入路口的道路，两者的 road_id
      通常不同，不能互相作等值过滤。相关性直接把每盏灯的停止点投影到 Agent 规划的同一车道上，并排除
      ego 后方停止点。空结果仍写 valid=false，使离线端能区分“新方法判定无灯”和“旧数据缺少新字段”。
"""

import math

import carla
import numpy as np

from worker.geometry import transform_to_list


_TRAFFIC_LIGHT_STATE_NAMES = {
    int(carla.TrafficLightState.Red): "red",
    int(carla.TrafficLightState.Yellow): "yellow",
    int(carla.TrafficLightState.Green): "green",
    int(carla.TrafficLightState.Off): "off",
    int(carla.TrafficLightState.Unknown): "unknown",
}
_SOURCE = "carla_stop_waypoint_route_v2"


def _lane_key(waypoint):
    return (int(waypoint.road_id), int(waypoint.section_id), int(waypoint.lane_id))


def _waypoint_metadata(waypoint):
    """把 CARLA waypoint 转成跨解释器可序列化的车道控制点。"""
    return {
        "road_id": int(waypoint.road_id),
        "section_id": int(waypoint.section_id),
        "lane_id": int(waypoint.lane_id),
        "s": float(waypoint.s),
        "transform": transform_to_list(waypoint.transform),
        "lane_width": float(waypoint.lane_width),
    }


def traffic_light_metadata(traffic_lights):
    """提取交通灯静态元数据、受控车道与 CARLA 原生停止点。"""
    metadata = []
    for light in traffic_lights:
        transform = light.get_transform()
        trigger = light.trigger_volume
        trigger_location = carla.Location(
            x=trigger.location.x, y=trigger.location.y, z=trigger.location.z)
        transform.transform(trigger_location)
        metadata.append({
            "id": int(light.id),
            "opendrive_id": str(light.get_opendrive_id()),
            "pole_index": int(light.get_pole_index()),
            "transform": transform_to_list(transform),
            "trigger_location": [trigger_location.x, trigger_location.y, trigger_location.z],
            "trigger_extent": [trigger.extent.x, trigger.extent.y, trigger.extent.z],
            "affected_lane_waypoints": [
                _waypoint_metadata(waypoint)
                for waypoint in light.get_affected_lane_waypoints()
            ],
            "stop_waypoints": [
                _waypoint_metadata(waypoint)
                for waypoint in light.get_stop_waypoints()
            ],
        })
    return metadata


def traffic_light_states(traffic_lights):
    """读取当前仿真帧全部交通灯状态，保持 actor ID 稳定排序。"""
    state_codes = [int(light.state) for light in traffic_lights]
    return [{"id": int(light.id), "state": _TRAFFIC_LIGHT_STATE_NAMES.get(code, "unknown"),
             "state_code": code}
            for light, code in zip(traffic_lights, state_codes)]


def _route_geometry(ego, agent):
    """把 ego 当前车道与 LocalPlanner 剩余队列拼成带车道键的世界系路线。"""
    ego_location = ego.get_location()
    ego_waypoint = ego.get_world().get_map().get_waypoint(ego_location)
    plan = list(agent.get_local_planner().get_plan())
    plan_waypoints = [item[0] for item in plan]
    waypoints = [ego_waypoint] + plan_waypoints
    route_xy = np.array(
        [[ego_location.x, ego_location.y]]
        + [[waypoint.transform.location.x, waypoint.transform.location.y]
           for waypoint in plan_waypoints],
        dtype=np.float64)
    lane_keys = [_lane_key(waypoint) for waypoint in waypoints]
    vectors = np.diff(route_xy, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    return route_xy, lane_keys, vectors, lengths, cumulative


def _project_stop(stop, route_xy, lane_keys, vectors, lengths, cumulative):
    """把停止点投到属于同一受控车道的规划线段，返回路线弧长；不相交则返回 None。"""
    key = (stop["road_id"], stop["section_id"], stop["lane_id"])
    segment_indices = np.array([
        index for index in range(len(vectors))
        if lane_keys[index] == key or lane_keys[index + 1] == key
    ], dtype=np.int64)
    if segment_indices.size == 0:
        return None

    valid = lengths[segment_indices] > np.finfo(np.float64).eps
    segment_indices = segment_indices[valid]
    if segment_indices.size == 0:
        return None
    starts = route_xy[segment_indices]
    segment_vectors = vectors[segment_indices]
    segment_lengths = lengths[segment_indices]
    point = np.asarray(stop["transform"][:2], dtype=np.float64)
    along = np.clip(
        ((point - starts) * segment_vectors).sum(1) / segment_lengths ** 2, 0.0, 1.0)
    closest = starts + along[:, None] * segment_vectors
    nearest = int(np.argmin(np.linalg.norm(closest - point, axis=1)))
    if np.linalg.norm(closest[nearest] - point) > stop["lane_width"] * 0.5:
        return None
    index = int(segment_indices[nearest])
    return float(cumulative[index] + along[nearest] * lengths[index])


def _candidate(light, stop_index, stop, route, ego_xy, state):
    """构造一个位于 ego 前方且规划确实经过的 CARLA 停止点候选。"""
    yaw = math.radians(float(stop["transform"][5]))
    forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    stop_xy = np.asarray(stop["transform"][:2], dtype=np.float64)
    if float(np.dot(stop_xy - ego_xy, forward)) <= 0.0:
        return None
    route_distance = _project_stop(stop, *route)
    if route_distance is None:
        return None
    return {
        "valid": True,
        "source": _SOURCE,
        "id": light["id"],
        "opendrive_id": light["opendrive_id"],
        "stop_waypoint_index": stop_index,
        "stop_location": stop["transform"][:3],
        "stop_yaw": stop["transform"][5],
        "lane_width": stop["lane_width"],
        "route_distance": route_distance,
        "state": state["state"],
        "state_code": state["state_code"],
    }


def relevant_traffic_control(ego, agent, metadata, states):
    """选择 Agent 当前规划上最先到达的 CARLA 原生交通灯停止点。

    参数:
        ego/agent: CARLA 主车与当前 BehaviorAgent
        metadata:  traffic_light_metadata() 的场景级结果
        states:    traffic_light_states() 的当前帧结果
    返回:
        可 JSON 序列化的 dict；无相关灯时为 `{"valid": False, "source": ...}`。
    """
    route = _route_geometry(ego, agent)
    if len(route[0]) < 2:
        return {"valid": False, "source": _SOURCE}
    ego_xy = route[0][0]
    state_by_id = {item["id"]: item for item in states}
    candidates = []
    for light in metadata:
        state = state_by_id[light["id"]]
        candidates.extend(
            candidate for candidate in (
                _candidate(light, index, stop, route, ego_xy, state)
                for index, stop in enumerate(light["stop_waypoints"])
            )
            if candidate is not None
        )
    return min(candidates, key=lambda item: item["route_distance"]) \
        if candidates else {"valid": False, "source": _SOURCE}
