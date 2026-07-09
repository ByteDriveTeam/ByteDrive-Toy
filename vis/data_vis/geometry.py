"""纯 numpy 复刻 CARLA 坐标变换与 3D->2D 投影（vis 侧不依赖 carla）。

模块: vis/data_vis/geometry.py
依赖: numpy, vis.data_vis.geometry_checks
读取配置: —（输入均为已解析的位姿/内参纯数据，自身不读 config）
对外接口:
    - transform_matrix(pose6) -> (4,4)              # CARLA 位姿 [x,y,z,roll,pitch,yaw](米/度) 的齐次矩阵
    - intrinsic_matrix(intr) -> (3,3)               # 由 fx,fy,cx,cy 组装针孔内参 K
    - world_to_camera(ego_pose6, cam_extrinsic6) -> (4,4)   # 世界->相机传感器系
    - world_to_ego(ego_pose6) -> (4,4)              # 世界->主车系
    - bbox_corners(bbox) -> (8,3)                   # 世界系下包围框 8 角点
    - project_points(pts_world, w2c, K) -> (uv(N,2), depth(N,))   # 投影到像素 + 相机前向深度
    - transform_points(pts, mat4) -> (N,3)          # 齐次批量变换
说明: 严格对齐 CARLA 客户端 Transform.get_matrix 的旋转约定（左手系，x 前 y 右 z 上）。相机投影时把
      传感器系 (x前,y右,z上) 换到标准像平面系 (x右,y下,z前) 再乘 K，与 CARLA 官方 bbox 投影示例一致。
      全部向量化：8 角点 / N 个 lidar 点一次矩阵乘出（规范 §9）。
"""

import numpy as np

from vis.data_vis.geometry_checks import check_pose6, check_points

# 包围框 8 角点在局部框系的符号组合：前 4 个为 -z 底面，后 4 个为 +z 顶面
_CORNER_SIGNS = np.array([
    (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
    (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
], dtype=np.float64)


def transform_matrix(pose6):
    """由 [x,y,z,roll,pitch,yaw]（米/度）构造 4x4 齐次变换，复刻 CARLA get_matrix。"""
    check_pose6(pose6)
    x, y, z, roll, pitch, yaw = pose6
    cr, sr = _cos_sin(roll)
    cp, sp = _cos_sin(pitch)
    cy, sy = _cos_sin(yaw)
    m = np.identity(4)
    m[:3, 3] = (x, y, z)
    # CARLA 左手系旋转矩阵（yaw 绕 z、pitch 绕 y、roll 绕 x 的特定组合）
    m[0, :3] = (cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr)
    m[1, :3] = (sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr)
    m[2, :3] = (sp, -cp * sr, cp * cr)
    return m


def intrinsic_matrix(intr):
    """由内参 dict(fx,fy,cx,cy) 组装针孔 K。"""
    return np.array([[intr["fx"], 0.0, intr["cx"]],
                     [0.0, intr["fy"], intr["cy"]],
                     [0.0, 0.0, 1.0]])


def world_to_camera(ego_pose6, cam_extrinsic6):
    """世界系 -> 相机传感器系：相机世界位姿 = ego ∘ 外参(主车局部)，取其逆。"""
    cam_world = transform_matrix(ego_pose6) @ transform_matrix(cam_extrinsic6)
    return np.linalg.inv(cam_world)


def world_to_ego(ego_pose6):
    """世界系 -> 主车系（俯视 BEV 把世界物体搬到以主车为原点的坐标）。"""
    return np.linalg.inv(transform_matrix(ego_pose6))


def transform_points(pts, mat4):
    """对 (N,3) 点批量施加 4x4 齐次变换，返回 (N,3)。"""
    pts = np.asarray(pts, dtype=np.float64)
    homo = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    return (homo @ mat4.T)[:, :3]


def bbox_corners(bbox):
    """由 location/extent/rotation 算世界系 8 角点（顺序同 _CORNER_SIGNS）。"""
    extent = np.asarray(bbox["extent"], dtype=np.float64)
    roll, pitch, yaw = bbox["rotation"]
    loc = bbox["location"]
    box_to_world = transform_matrix([loc[0], loc[1], loc[2], roll, pitch, yaw])
    local = _CORNER_SIGNS * extent  # (8,3) 局部角点
    return transform_points(local, box_to_world)


def project_points(pts_world, w2c, K):
    """把世界系点投影到像素坐标，并返回相机前向深度（米）。

    参数:
        pts_world: (N,3) 世界坐标点
        w2c:       (4,4) 世界->相机传感器系矩阵
        K:         (3,3) 针孔内参
    返回:
        uv:    (N,2) 像素坐标（可能落在画面外，由调用方裁剪）
        depth: (N,) 相机前向距离；<=0 表示在相机后方
    """
    check_points(pts_world)
    cam = transform_points(pts_world, w2c)  # 传感器系 (x前,y右,z上)
    # 换到标准像平面系 (x右,y下,z前)：右=+y、下=-z、前=+x
    img = np.stack([cam[:, 1], -cam[:, 2], cam[:, 0]], axis=1)
    depth = img[:, 2]
    safe = np.where(np.abs(depth) < 1e-6, 1e-6, depth)  # 避免除零；后方点由 depth 符号过滤
    proj = img @ K.T
    uv = proj[:, :2] / safe[:, None]
    return uv, depth


def _cos_sin(deg):
    r = np.radians(deg)
    return np.cos(r), np.sin(r)
