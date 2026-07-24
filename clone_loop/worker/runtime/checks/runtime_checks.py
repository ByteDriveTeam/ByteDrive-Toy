def check_route(route):
    """校验对象: CarlaRuntime.reset.route —— 起终点须为完整 CARLA 位姿。"""
    if not isinstance(route, dict) or "start" not in route or "end" not in route:
        raise ValueError("闭环路线必须包含 start/end")
    if len(route["start"]) != 6 or len(route["end"]) != 6:
        raise ValueError("闭环路线 start/end 均期望六维 CARLA 位姿")


def check_control(control):
    """校验对象: CarlaRuntime.step.control —— 执行器字段完整且落在 CARLA 范围。"""
    expected = {"throttle", "steer", "brake"}
    if not expected.issubset(control):
        raise ValueError("闭环控制缺少字段: {}".format(sorted(expected.difference(control))))
    if not 0 <= float(control["throttle"]) <= 1 \
            or not -1 <= float(control["steer"]) <= 1 \
            or not 0 <= float(control["brake"]) <= 1:
        raise ValueError("闭环控制超出 CARLA 执行器范围")


def check_blueprints(blueprints, pattern):
    """校验对象: ego/traffic.vehicle_filter —— 过滤结果不得为空。"""
    if not blueprints:
        raise RuntimeError("找不到车辆蓝图: {}".format(pattern))
