"""把 HDMap、bbox 与交通控制标注合成为驾驶相关 BEVSeg。

模块: data/bevseg_synthesis/bevseg_synthesis.py
依赖: cv2, numpy, data.hd_map, data.driving_targets, vis.data_vis.geometry
读取配置: data.bevseg.extent_m/resolution/layers/map_dir/map_name_template/lane_types/
          lane_layer_map/box_layers/state_layers/lane_half_width_m/line_width_m/
          unknown_lane_class/stop_line_width_m
对外接口:
    - BevSegRasterizer(cfg) -> object
      .rasterize_frame(scene_meta, frame_meta, map_obj, current_pose, reference_pose=None)
        -> dict[str, ndarray]
      .rasterize_frames(scene_meta, frame_metas, map_obj, current_pose, reference_pose=None)
        -> dict[str, ndarray]
说明: 公共坐标契约为 X=右、Y=前；内部复用项目既有 world_to_ego 数值变换，避免改变 DrivingModel。
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from data.driving_targets import BevParams, ego_xy_to_pixel
from data.hd_map import HdMap
from vis.data_vis.geometry import bbox_corners, transform_points, world_to_ego


__all__ = ["BevSegRasterizer"]


class BevSegRasterizer:
    """按统一的 ego-BEV 坐标契约栅格化驾驶相关语义。"""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        extent_m = float(cfg.extent_m)
        resolution = int(cfg.resolution)
        self.bev = BevParams(-extent_m * 0.5, extent_m * 0.5,
                             -extent_m * 0.5, extent_m * 0.5,
                             resolution, resolution)
        self.layers = tuple(cfg.layers)
        self.layer_index = {name: index for index, name in enumerate(self.layers)}
        map_dir = Path(cfg.map_dir)
        self.map_dir = (map_dir if map_dir.is_absolute()
                        else Path(__file__).resolve().parents[2] / map_dir)
        self.lane_types = dict(cfg.lane_types)
        self.lane_layer_map = {int(k): v for k, v in cfg.lane_layer_map.items()}
        self.box_layers = dict(cfg.box_layers)
        self.state_layers = dict(cfg.state_layers)

    def load_map(self, scene_meta):
        """按场景地图名加载 HDMap。"""
        map_name = scene_meta.get("map")
        if not map_name:
            raise KeyError("scene meta 缺少 map")
        map_name = str(map_name).replace("_Opt", "")
        path = self.map_dir / str(self.cfg.map_name_template).format(map=map_name)
        return HdMap(path)

    def rasterize_frame(self, scene_meta, frame_meta, map_obj, current_pose,
                        reference_pose=None):
        """生成一帧二值语义层与车道 sin/cos 方向。"""
        batch = self.rasterize_frames(
            scene_meta, [frame_meta], map_obj, current_pose, reference_pose)
        return {name: value[0] for name, value in batch.items()}

    def rasterize_frames(self, scene_meta, frame_metas, map_obj, current_pose,
                         reference_pose=None):
        """批量生成同一 ego 坐标系下的多帧栅格，并复用静态 HDMap 投影。"""
        pose = np.asarray(current_pose, dtype=np.float64)
        reference = pose if reference_pose is None else np.asarray(reference_pose, dtype=np.float64)
        drivable, lane_class, lane_direction = map_obj.drivable_lane_bev(
            pose.tolist(), self.bev, self.cfg.lane_half_width_m, self.cfg.line_width_m,
            self.lane_types, self.cfg.unknown_lane_class)
        semantic_base = np.zeros(
            (len(self.layers), self.bev.height, self.bev.width), dtype=np.float32)
        direction_base = np.zeros((2, self.bev.height, self.bev.width), dtype=np.float32)
        semantic_base[self.layer_index["drivable"]] = drivable
        self._copy_lane_layers(
            semantic_base, direction_base, lane_class, lane_direction)
        self._rasterize_boxes(
            semantic_base, scene_meta.get("static_bboxes", []), reference, exclude_ego=False)
        semantic = np.repeat(semantic_base[None], len(frame_metas), axis=0)
        direction = np.repeat(direction_base[None], len(frame_metas), axis=0)
        for frame_semantic, frame_meta in zip(semantic, frame_metas):
            self._rasterize_boxes(
                frame_semantic, frame_meta.get("bboxes", []), reference, exclude_ego=True)
            self._rasterize_control(
                frame_semantic, frame_meta, scene_meta, map_obj, reference)
        return {"semantic": semantic, "direction": direction}

    def _copy_lane_layers(self, semantic, direction, lane_class, lane_direction):
        for class_id, target_name in self.lane_layer_map.items():
            if target_name not in self.layer_index:
                continue
            mask = lane_class == class_id
            semantic[self.layer_index[target_name]][mask] = 1.0
        lane_mask = lane_class > 0
        # 既有 HDMap 方向为 (前向, 右向)，公共契约为 (右向, 前向)。
        direction[0] = np.where(lane_mask, lane_direction[1], 0.0)
        direction[1] = np.where(lane_mask, lane_direction[0], 0.0)

    def _rasterize_boxes(self, semantic, boxes, current_pose, exclude_ego):
        w2e = world_to_ego(current_pose)
        for box in boxes:
            label = box.get("semantic")
            if exclude_ego and label == "ego":
                continue
            target = self.box_layers.get(label)
            if target not in self.layer_index:
                continue
            corners = transform_points(bbox_corners(box), w2e)
            rows, cols = ego_xy_to_pixel(corners[:, :2], self.bev)
            points = np.stack((cols, rows), axis=1).round().astype(np.int32)
            polygon = cv2.convexHull(points)
            cv2.fillConvexPoly(semantic[self.layer_index[target]], polygon, 1.0)

    def _rasterize_control(self, semantic, frame_meta, scene_meta, map_obj, current_pose):
        control = frame_meta.get("relevant_traffic_control")
        if not control or not control.get("valid", False):
            return
        state = control.get("state")
        target = self.state_layers.get(state)
        if target not in self.layer_index:
            return
        center = np.asarray(control.get("stop_location", [0, 0, 0]), dtype=np.float64)
        center = transform_points(center[None], world_to_ego(current_pose))[0, :2]
        yaw = math.radians(float(control.get("stop_yaw", 0.0)))
        world_direction = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
        ego_direction = world_direction @ world_to_ego(current_pose)[:2, :2].T
        ego_direction /= np.linalg.norm(ego_direction).clip(1e-6)
        # 既有数值顺序为 (前, 右)，停止线沿行驶方向的法向展开。
        normal = np.array([-ego_direction[1], ego_direction[0]])
        half_width = float(control.get("lane_width", self.cfg.stop_line_width_m)) * 0.5
        endpoints = center + np.array([-half_width, half_width])[:, None] * normal
        rows, cols = ego_xy_to_pixel(endpoints, self.bev)
        points = np.stack((cols, rows), axis=1).round().astype(np.int32)
        pixels_per_meter = self.bev.width / (self.bev.y_max - self.bev.y_min)
        cv2.line(semantic[self.layer_index[target]], tuple(points[0]), tuple(points[1]), 1.0,
                 max(int(round(self.cfg.stop_line_width_m * pixels_per_meter)), 1))
