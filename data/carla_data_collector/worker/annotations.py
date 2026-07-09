"""带语义的包围框抽取：动态 actor（逐帧）与静态环境物体（每场景一次）。

模块: worker/annotations.py
依赖: carla, worker.geometry
读取配置: —（输入为 carla world/actor，自身不读 config）
对外接口:
    - dynamic_bboxes(world, ego_id) -> list[dict]   # 车辆/行人/主车的世界系包围框（逐帧）
    - static_bboxes(world) -> list[dict]            # 交通标志/信号灯/杆/静态物等环境包围框（每场景）
说明: Design ⑫ 要求包围框带语义。动态物每帧抽取（会移动）；静态环境物只在场景开始抽一次（不动），
      由 collect.collect_chunk/prepare_drive 分别放入帧索引与场景级元数据，避免重复计算。
"""

import carla

from worker.geometry import bbox_to_dict

# 采集的静态环境语义类别（carla.CityObjectLabel）；标签字符串即语义
_ENV_LABELS = [
    ("traffic_sign", carla.CityObjectLabel.TrafficSigns),
    ("traffic_light", carla.CityObjectLabel.TrafficLight),
    ("pole", carla.CityObjectLabel.Poles),
    ("static", carla.CityObjectLabel.Static),
]


def static_bboxes(world):
    """抽取关心类别的环境物体世界系包围框。"""
    return [bbox_to_dict(bb, semantic, "env")
            for semantic, label in _ENV_LABELS
            for bb in world.get_level_bbs(label)]


def _actor_bbox(actor, semantic):
    """把 actor 的局部包围框换算到世界系并打成 dict。"""
    bb = actor.bounding_box
    tf = actor.get_transform()
    center = carla.Location(bb.location.x, bb.location.y, bb.location.z)
    tf.transform(center)  # 原地变换到世界坐标
    world_bb = carla.BoundingBox(center, bb.extent)
    world_bb.rotation = tf.rotation
    return bbox_to_dict(world_bb, semantic, "actor", actor.id)


def dynamic_bboxes(world, ego_id):
    """抽取全部车辆与行人的世界系包围框；主车单独标注语义 ego。"""
    actors = world.get_actors()
    vehicles = [_actor_bbox(a, "ego" if a.id == ego_id else "vehicle")
                for a in actors.filter("vehicle.*")]
    walkers = [_actor_bbox(a, "pedestrian") for a in actors.filter("walker.pedestrian.*")]
    return vehicles + walkers
