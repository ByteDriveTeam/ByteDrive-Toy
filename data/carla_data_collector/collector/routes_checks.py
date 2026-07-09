# 本文件为 collector/routes.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_spawn_points(spawn_points):
    """校验对象: build_route_queue 入参 spawn_points —— 至少两点、每点含 x/y/z。

    少于两点无法构成任何起终点组合；距离区间合法性已由 schema 在加载期拦截。
    """
    assert len(spawn_points) >= 2, "可达点不足两个，无法构建路线队列"
    assert all(len(p) >= 3 for p in spawn_points), "每个可达点至少需含 [x,y,z]"
