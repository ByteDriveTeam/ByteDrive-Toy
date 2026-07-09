"""CARLA 语义标签到颜色的调色板与向量化映射（用于 lidar 点按语义着色）。

模块: vis/data_vis/palette.py
依赖: numpy
读取配置: —（CARLA 官方 CityScapes 调色板是领域常量，非可调配置，故不进 config）
对外接口:
    - tag_to_bgr(tags) -> np.ndarray(N,3) uint8     # 语义标签数组 -> 逐点 BGR
说明: 取值对齐 CARLA 0.9.15 语义分割/语义雷达的 obj_tag（CityScapesPalette）。表内按 RGB 记录便于和
      官方文档逐行核对，映射时再翻成 OpenCV 的 BGR。越界标签按取模回绕，保证不抛错。
"""

import numpy as np

# 索引即 obj_tag；值为官方 RGB（见 CARLA 语义分割相机文档）
_TAG_RGB = np.array([
    (0, 0, 0),        # 0  Unlabeled
    (70, 70, 70),     # 1  Building
    (100, 40, 40),    # 2  Fence
    (55, 90, 80),     # 3  Other
    (220, 20, 60),    # 4  Pedestrian
    (153, 153, 153),  # 5  Pole
    (157, 234, 50),   # 6  RoadLine
    (128, 64, 128),   # 7  Road
    (244, 35, 232),   # 8  SideWalk
    (107, 142, 35),   # 9  Vegetation
    (0, 0, 142),      # 10 Vehicles
    (102, 102, 156),  # 11 Wall
    (220, 220, 0),    # 12 TrafficSign
    (70, 130, 180),   # 13 Sky
    (81, 0, 81),      # 14 Ground
    (150, 100, 100),  # 15 Bridge
    (230, 150, 140),  # 16 RailTrack
    (180, 165, 180),  # 17 GuardRail
    (250, 170, 30),   # 18 TrafficLight
    (110, 190, 160),  # 19 Static
    (170, 120, 50),   # 20 Dynamic
    (45, 60, 150),    # 21 Water
    (145, 170, 100),  # 22 Terrain
], dtype=np.uint8)

# 预翻为 BGR，映射时直接索引（避免每次调用再反转通道）
_TAG_BGR = _TAG_RGB[:, ::-1].copy()


def tag_to_bgr(tags):
    """把语义标签数组映射为逐点 BGR 颜色。

    参数:
        tags: 形状 (N,) 的整型标签（CARLA obj_tag）
    返回:
        (N, 3) uint8 的 BGR 颜色数组
    """
    return _TAG_BGR[np.asarray(tags).astype(np.intp) % len(_TAG_BGR)]
