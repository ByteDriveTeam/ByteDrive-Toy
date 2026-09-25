"""把五帧三目驾驶样本的输入、独立占用与规划监督合成一张检查图。

模块: vis/data_vis/driving_sample/driving_sample.py
依赖: cv2, numpy, vis.driving_vis.render, vis.data_vis.driving_sample.checks
读取配置: data_vis.driving.tile_width_px/tile_height_px, driving_vis.lane_map.*, model.driving.bev.*
对外接口:
    - render_driving_sample(sample, cfg, scene_name, frame_idx) -> np.ndarray
说明: 所有 BEV 面板沿用数据集的远端在上、车辆前向为 X、右侧为 Y 的轴约定。
"""

from __future__ import annotations

import cv2
import numpy as np

from vis.data_vis.driving_sample.checks.driving_sample_checks import check_driving_sample
from vis.driving_vis.render import colorize_lane_map

__all__ = ["render_driving_sample"]

_BG = (23, 25, 30)
_SCENE = (90, 200, 90)
_AGENT = (75, 150, 255)
_VISIBLE = (190, 190, 75)
_EGO = (0, 255, 255)


def _array(value):
    return value.detach().cpu().numpy()


def _camera(tensor):
    return np.ascontiguousarray(_array(tensor).transpose(1, 2, 0))


def _binary(mask, color):
    mask = np.asarray(mask, dtype=bool)
    image = np.full((*mask.shape, 3), _BG, dtype=np.uint8)
    image[mask] = color
    return image


def _xy_pixel(xy, shape, bev):
    height, width = shape[:2]
    x = np.asarray(xy)[..., 0]
    y = np.asarray(xy)[..., 1]
    row = (bev.x_max_m - x) * height / (bev.x_max_m - bev.x_min_m)
    col = (y - bev.y_min_m) * width / (bev.y_max_m - bev.y_min_m)
    return np.stack((col, row), -1).round().astype(np.int32)


def _path(image, xy, valid, bev, color):
    points = np.asarray(xy)[np.asarray(valid, dtype=bool)]
    if not len(points):
        return image
    canvas = image.copy()
    pixels = _xy_pixel(np.concatenate((np.zeros((1, 2)), points[:, :2])), canvas.shape, bev)
    cv2.polylines(canvas, [pixels], False, color, 2, cv2.LINE_AA)
    for point in pixels[1:]:
        cv2.circle(canvas, tuple(point), 2, color, -1, cv2.LINE_AA)
    return canvas


def _boxes(image, sample, bev, draw_future):
    canvas = image.copy()
    classes = _array(sample["detect_class"])
    boxes = _array(sample["detect_box"])
    futures = _array(sample["detect_future"])
    future_valid = _array(sample["detect_future_valid"])
    for index in np.flatnonzero(classes >= 0):
        cx, cy, _z, length, width, _height, yaw, _vx, _vy = boxes[index]
        offsets = np.array([[length, width], [length, -width],
                            [-length, -width], [-length, width]], dtype=np.float32) * .5
        rotation = np.array([[np.cos(yaw), -np.sin(yaw)],
                             [np.sin(yaw), np.cos(yaw)]], dtype=np.float32)
        corners = offsets @ rotation.T + (cx, cy)
        cv2.polylines(canvas, [_xy_pixel(corners, canvas.shape, bev)], True,
                      _SCENE if classes[index] == 0 else _AGENT, 2, cv2.LINE_AA)
        if draw_future:
            valid = future_valid[index]
            if valid.any():
                positions = np.concatenate((boxes[index:index + 1, :2], futures[index, valid]), 0)
                cv2.polylines(canvas, [_xy_pixel(positions, canvas.shape, bev)], False,
                              _SCENE if classes[index] == 0 else _AGENT, 1, cv2.LINE_AA)
    return canvas


def _tile(image, title, width, height):
    canvas = np.full((height, width, 3), _BG, dtype=np.uint8)
    usable_height = height - 24
    scale = min(width / image.shape[1], usable_height / image.shape[0])
    size = (max(1, int(round(image.shape[1] * scale))),
            max(1, int(round(image.shape[0] * scale))))
    resized = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)
    left = (width - size[0]) // 2
    top = 24 + (usable_height - size[1]) // 2
    canvas[top:top + size[1], left:left + size[0]] = resized
    cv2.putText(canvas, title, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, .46,
                (235, 235, 235), 1, cv2.LINE_AA)
    return canvas


def _summary(sample, scene_name, frame_idx):
    scene = _array(sample["scene_occ"])
    agent = _array(sample["agent_occ"])
    detected = _array(sample["detect_class"])
    valid = _array(sample["traj_valid"])
    lines = ("{} / f{}    scene voxels: {}    Agent voxels: {}    Detect: {}".format(
                 scene_name, frame_idx, int(scene.sum()), int(agent.sum()),
                 int((detected >= 0).sum())),
             "ego GT: {}/{}    target XY: {:.1f}, {:.1f}m".format(
                 int(valid.sum()), len(valid), *_array(sample["target_point"])))
    return lines


def _header(sample, scene_name, frame_idx, width):
    canvas = np.full((56, width, 3), _BG, dtype=np.uint8)
    lines = _summary(sample, scene_name, frame_idx)
    for index, line in enumerate(lines):
        cv2.putText(canvas, line, (12, 21 + 25 * index), cv2.FONT_HERSHEY_SIMPLEX,
                    .56, (235, 235, 235), 1, cv2.LINE_AA)
    return canvas


def render_driving_sample(sample, cfg, scene_name, frame_idx):
    """渲染真实 DrivingDataset 单样本，保留场景与 Agent 占用的独立视图。"""
    check_driving_sample(sample)
    bev = cfg.model.driving.bev
    width = cfg.data_vis.driving.tile_width_px
    height = cfg.data_vis.driving.tile_height_px
    cameras = tuple(cfg.data.driving.cameras)
    front = cameras.index("front")
    history_valid = _array(sample["history_valid"])
    history = [(_camera(sample["history_rgb"][step, front]),
                "front t-{}{}".format(4 - step, " (pad)" if not history_valid[step] else ""))
               for step in range(4)]
    current = [(_camera(sample["rgb"][cameras.index(camera)]), camera + " t0")
               for camera in ("front_left", "front", "front_right")]
    lidar = _array(sample["lidar_occupied"])[0].any(axis=0)
    current.append((_binary(lidar, _VISIBLE), "LiDAR occupancy"))

    drivable = _binary(_array(sample["drivable"]) > .5, _SCENE)
    lane_vis = cfg.driving_vis.lane_map
    lane = colorize_lane_map(
        _array(sample["lane_class"]),
        _array(sample["lane_direction"]) * _array(sample["lane_direction_valid"])[None],
        lane_vis.class_colors, lane_vis.arrow_color, lane_vis.arrow_stride_px,
        lane_vis.arrow_length_px, lane_vis.arrow_thickness, lane_vis.arrow_tip_ratio)
    stop = _binary(_array(sample["stop_line"]) > .5, (50, 60, 240))
    ego_path = _path(drivable, _array(sample["trajectory"]),
                     _array(sample["traj_valid"]), bev, _EGO)
    target = _xy_pixel(_array(sample["target_point"])[None], ego_path.shape, bev)[0]
    cv2.circle(ego_path, tuple(target), 4, (255, 100, 255), -1, cv2.LINE_AA)
    fields = [(drivable, "drivable"), (lane, "lane semantic / dir"),
              (stop, "stop line"), (ego_path, "ego 6s GT / target")]

    scene = _array(sample["scene_occ"]).astype(bool)
    agent = _array(sample["agent_occ"]).astype(bool)
    masks = [(_binary(scene.any(axis=0), _SCENE), "scene 3D max"),
             (_binary(agent.any(axis=0), _AGENT), "Agent 3D max"),
             (_binary(_array(sample["scene_occ_mask"]).any(axis=0), _VISIBLE), "scene visibility"),
             (_binary(_array(sample["agent_occ_mask"]).any(axis=0), _VISIBLE), "Agent visibility")]
    scene_slices = [(_binary(part.any(axis=0), _SCENE), "scene z " + str(index + 1) + "/3")
                    for index, part in enumerate(np.array_split(scene, 3, axis=0))]
    agent_slices = [(_binary(part.any(axis=0), _AGENT), "Agent z " + str(index + 1) + "/3")
                    for index, part in enumerate(np.array_split(agent, 3, axis=0))]
    box_base = _boxes(_binary(agent.any(axis=0), _AGENT), sample, bev, False)
    future_base = _boxes(_binary(agent.any(axis=0), _AGENT), sample, bev, True)
    rows = (history, current, fields, masks,
            scene_slices + [(box_base, "Detect boxes")],
            agent_slices + [(future_base, "Agent future 3s")])
    grid = np.concatenate([np.concatenate([_tile(image, title, width, height)
                                            for image, title in row], axis=1)
                           for row in rows], axis=0)
    return np.concatenate((_header(sample, scene_name, frame_idx, grid.shape[1]), grid), axis=0)
