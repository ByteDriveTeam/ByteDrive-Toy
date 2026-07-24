def check_route_trace(trace):
    """校验对象: RouteNavigator 全局规划结果 —— 至少需要两个路线点。"""
    if trace is None or len(trace) < 2:
        raise RuntimeError("CARLA 全局规划未生成有效路线")
