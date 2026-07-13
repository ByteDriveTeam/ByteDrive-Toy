"""可视化着色：CARLA 语义标签调色板 + 光流/速度场 HSV 配色，data_vis 与 pred_vis 共用。

模块: vis/data_vis/palette/palette.py
依赖: numpy, cv2
读取配置: —（CARLA 官方 CityScapes 调色板是领域常量，非可调配置，故不进 config）
对外接口:
    - tag_to_bgr(tags) -> np.ndarray(N,3) uint8         # 语义标签数组 -> 逐点 BGR
    - flow_to_bgr(vx, vy, max_mag) -> np.ndarray uint8  # 光流/速度分量 -> 经典光流配色 BGR
说明: 取值对齐 CARLA 0.9.15 语义分割/语义雷达的 obj_tag（CityScapesPalette）。表内按 RGB 记录便于和
      官方文档逐行核对，映射时再翻成 OpenCV 的 BGR。越界标签按取模回绕，保证不抛错。
      flow_to_bgr 为原始光流与解码速度共用的唯一着色实现（避免两套 vis 复制后走样），仅逐元素运算故布局无关。
"""

import cv2
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


def flow_to_bgr(vx, vy, max_mag):
    """光流/速度矢量场 -> 经典光流配色 BGR：色相编码方向、亮度编码幅值。

    参数:
        vx, vy:  同形状的分量数组（vx=水平、vy=竖直）；仅逐元素运算，故 (H,W)/展平等布局皆可
        max_mag: 满亮度基准幅值；>0 以之为准，=0 则按本帧幅值 99 分位自适应（免去对单位的先验假设）
    返回:
        形状为 vx.shape + (3,) 的 BGR uint8
    """
    vx = vx.astype(np.float32)
    vy = vy.astype(np.float32)
    mag = np.sqrt(vx * vx + vy * vy)
    ang = np.arctan2(vy, vx)  # [-pi, pi]
    denom = max_mag if max_mag > 0 else float(np.percentile(mag, 99)) + 1e-6
    hsv = np.empty(vx.shape + (3,), dtype=np.uint8)
    hsv[..., 0] = np.minimum((ang + np.pi) * (90.0 / np.pi), 179.0).astype(np.uint8)  # 方向->色相
    hsv[..., 1] = 255
    hsv[..., 2] = (np.clip(mag / denom, 0.0, 1.0) * 255.0).astype(np.uint8)            # 幅值->亮度
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
