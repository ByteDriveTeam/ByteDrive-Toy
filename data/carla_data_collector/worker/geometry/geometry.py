"""carla 几何对象与纯数值之间的转换，以及相机内参推导。

模块: worker/geometry/geometry.py
依赖: math, carla
读取配置: —（输入为已解析的标量/配置片段，自身不读 config）
对外接口:
    - transform_to_list(transform) -> list[6]          # [x,y,z,roll,pitch,yaw]
    - location_to_list(location) -> list[3]            # [x,y,z]
    - bbox_to_dict(bbox, semantic, source, actor_id=None) -> dict
    - make_transform(pose6) -> carla.Transform         # 由 [x,y,z,roll,pitch,yaw] 构造
    - compute_intrinsics(width, height, fov_deg) -> dict   # 针孔内参 fx,fy,cx,cy
说明: 把 carla 专有类型在 worker 边界上转成可 JSON 序列化的纯数据，便于经控制管道回传。
"""

import math

import carla


def transform_to_list(transform):
    """carla.Transform -> [x,y,z,roll,pitch,yaw]（米/度）。"""
    loc, rot = transform.location, transform.rotation
    return [loc.x, loc.y, loc.z, rot.roll, rot.pitch, rot.yaw]


def location_to_list(location):
    return [location.x, location.y, location.z]


def make_transform(pose6):
    """[x,y,z,roll,pitch,yaw] -> carla.Transform。"""
    x, y, z, roll, pitch, yaw = pose6
    return carla.Transform(carla.Location(x=x, y=y, z=z),
                           carla.Rotation(roll=roll, pitch=pitch, yaw=yaw))


def bbox_to_dict(bbox, semantic, source, actor_id=None):
    """把一个包围框（carla.BoundingBox，世界坐标）连同语义打成 dict。"""
    return {
        "semantic": semantic,        # 语义标签字符串
        "source": source,            # "actor" 或 "env"
        "id": actor_id,              # actor 来源时为其 id，否则 None
        "location": location_to_list(bbox.location),
        "extent": [bbox.extent.x, bbox.extent.y, bbox.extent.z],
        "rotation": [bbox.rotation.roll, bbox.rotation.pitch, bbox.rotation.yaw],
    }


def compute_intrinsics(width, height, fov_deg):
    """由分辨率与水平 FOV 推导针孔相机内参（方形像素，主点居中）。

    fx = W / (2·tan(fov/2))；RGB 与 Depth 共享同一内参（二者尺寸/FOV 相同）。
    """
    fx = width / (2.0 * math.tan(math.radians(fov_deg) * 0.5))
    return {"fx": fx, "fy": fx, "cx": width * 0.5, "cy": height * 0.5,
            "width": width, "height": height, "fov": fov_deg}
