def check_route_inputs(spawn_points, min_distance, max_distance, max_episodes):
    """校验对象: build_route_queue 入参 —— 生成点与距离区间须能构成路线。"""
    if len(spawn_points) < 2:
        raise ValueError("CARLA 地图至少需要两个推荐生成点")
    if not 0 < min_distance < max_distance:
        raise ValueError("路线距离需满足 0 < min_distance < max_distance")
    if max_episodes < 0:
        raise ValueError("max_episodes 必须 >= 0")
