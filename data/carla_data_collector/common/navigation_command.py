"""从 CARLA 路线生成五类导航指令；恢复使用独立并行通道。

模块: common/navigation_command.py
依赖: math
读取配置: —（前视距离和转角阈值由调用方传入）
对外接口:
    - navigation_command_from_route(route, index, lookahead_m, turn_threshold_deg) -> int
    - navigation_command_from_plan(location, plan, lookahead_m, turn_threshold_deg) -> int
    - lane_change_from_sequence(lane_keys, stable_points) -> int | None
说明: 导航为直行、左转、右转、右变道、左变道；恢复为无、向右、向左。
"""

import math


_OPTION_TO_COMMAND = {1: 1, 2: 2, 5: 4, 6: 3}
# Independent of the five mutually exclusive route commands.
RECOVER_RIGHT = 1
RECOVER_LEFT = 2


def lane_change_from_sequence(lane_keys, stable_points):
    """从连续稳定的道路/车道身份中找第一处同路同向相邻车道变化。"""
    previous = None
    for start in range(len(lane_keys) - stable_points + 1):
        key = lane_keys[start]
        if key is None or any(item != key for item in lane_keys[start:start + stable_points]):
            continue
        if previous is not None and key != previous:
            before, after = previous, key
            previous = key
            if before[0] == after[0] and before[1] * after[1] > 0 \
                    and abs(after[1] - before[1]) == 1:
                return 4 if after[1] > before[1] else 3
        else:
            previous = key
    return None


def navigation_command_from_route(route, index, lookahead_m, turn_threshold_deg):
    """按路线前视范围返回五类指令 ID；无明确选项时根据车道或几何推断。"""
    points, arc = route["points"], route["arc_m"]
    start = int(index)
    if len(points) == 0 or len(points) != len(arc) or not 0 <= start < len(points):
        raise ValueError("导航路线的 points/arc_m/index 无效")
    end = start + 1
    while end < len(arc) and arc[end] <= arc[start] + lookahead_m:
        end += 1
    options = route.get("road_options")
    if options is not None:
        for option in options[start:end]:
            command = _OPTION_TO_COMMAND.get(int(option))
            if command is not None:
                return command
    lane_keys = route.get("lane_keys")
    if lane_keys is not None:
        for before, after in zip(lane_keys[start:end - 1], lane_keys[start + 1:end]):
            if before[:2] == after[:2] and before[2] * after[2] > 0 \
                    and abs(before[2]) != abs(after[2]):
                return 4 if after[2] > before[2] else 3
    moving = next((j for j in range(start + 1, end)
                   if math.hypot(points[j][0] - points[start][0],
                                 points[j][1] - points[start][1]) > 1e-3), None)
    if moving is None or end - start < 3:
        return 0
    incoming = (points[moving][0] - points[start][0],
                points[moving][1] - points[start][1])
    outgoing = (points[end - 1][0] - points[start][0],
                points[end - 1][1] - points[start][1])
    angle = math.degrees(math.atan2(
        incoming[0] * outgoing[1] - incoming[1] * outgoing[0],
        incoming[0] * outgoing[0] + incoming[1] * outgoing[1]))
    return 2 if angle >= turn_threshold_deg else 1 if angle <= -turn_threshold_deg else 0


def navigation_command_from_plan(location, plan, lookahead_m, turn_threshold_deg):
    """把专家 LocalPlanner 剩余队列转为路线，再取前方导航指令。"""
    plan = list(plan)
    if not plan:
        return 0
    # A displaced ego must not turn a straight assigned route into a turn
    # command. Anchor the geometric fallback on the route's own tangent.
    first = plan[0][0].transform
    forward = first.get_forward_vector()
    along = min(0.0, (location.x - first.location.x) * forward.x +
                (location.y - first.location.y) * forward.y)
    points = [[float(first.location.x + along * forward.x),
               float(first.location.y + along * forward.y)]]
    arc = [0.0]
    options = [4]
    lane_keys = [None]
    for waypoint, option in plan:
        position = waypoint.transform.location
        point = [float(position.x), float(position.y)]
        arc.append(arc[-1] + math.hypot(point[0] - points[-1][0],
                                        point[1] - points[-1][1]))
        points.append(point)
        options.append(int(option))
        lane_keys.append([int(waypoint.road_id), int(waypoint.section_id),
                          int(waypoint.lane_id)])
        if arc[-1] > lookahead_m:
            break
    # 当前点缺少车道键；RoadOption 可用时仍优先选它。
    if len(lane_keys) > 1:
        lane_keys[0] = lane_keys[1]
    return navigation_command_from_route(
        {"points": points, "arc_m": arc, "road_options": options,
         "lane_keys": lane_keys}, 0, lookahead_m, turn_threshold_deg)
