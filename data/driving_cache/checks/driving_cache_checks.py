"""驾驶训练缓存中导航路线与类别的输入检查。"""

import torch


def check_navigation_route(route_geometry, route_index):
    """校验对象: navigation_command_from_route 的路线坐标和当前索引。"""
    points = torch.as_tensor(route_geometry["points"])
    arc = torch.as_tensor(route_geometry["arc_m"])
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2 \
            or arc.ndim != 1 or arc.shape[0] != points.shape[0] \
            or not 0 <= int(route_index) < points.shape[0]:
        raise ValueError("route_geometry 需要 points[N,>=2]、arc_m[N] 和有效 route_index")
    if not torch.isfinite(points).all() or not torch.isfinite(arc).all() \
            or not (arc[1:] >= arc[:-1]).all():
        raise ValueError("route_geometry 的点和累计里程必须有限且按路线递增")
    for name in ("road_options", "lane_keys"):
        if name in route_geometry and len(route_geometry[name]) != points.shape[0]:
            raise ValueError("route_geometry.{} 长度必须与 points 一致".format(name))
