"""在 CARLA 中执行纯轨迹模型控制并同步采集 10Hz 真值与 2Hz 传感器。

模块: worker/model_runtime/model_runtime.py
依赖: math, random, carla, numpy, clone_loop.shared_frame/worker.navigation,
      common.protocol, worker.actors/annotations/collect/session/traffic_control
读取配置:
    carla_collector.worker.command_timeout_s / simulation.fixed_delta_seconds / warmup_ticks /
        traffic.* / ego.vehicle_filter / cameras.* / lidar.* / collection.capture_every_n_ticks /
        collection.max_frames_per_scene / model_collection.*
    clone_loop.route.* / safety.max_route_deviation_m
    data.driving.cameras
对外接口:
    - ModelCollectionRuntime(client, cfg, allocator, frame, lidar)
        .start(map_name, seed, weather, route) -> dict
        .step(control) -> dict
        .flush_pending() -> dict
        .close() -> None
说明: 模型控制只接收主进程由 Winner 轨迹转换出的油门/转角/制动量；本模块不接收、
      不解释行为标签。模型输入以 10Hz 写固定共享区，训练数据中的 RGB/LiDAR 按配置 2Hz
      写 arena。arena 溢出帧会完整暂存，主进程落盘并 reset 后由 flush_pending 原样写入。
"""

import math
import random

import carla
import numpy as np

from clone_loop.worker.navigation import RouteNavigator
from common import protocol as P
from common.shm import ArenaFull
from worker import actors, annotations, collect, session, traffic_control
from worker.geometry import compute_intrinsics
from worker.sensors import SensorRig


__all__ = ["ModelCollectionRuntime"]

_LIDAR_DTYPE = np.dtype(P.SEMANTIC_LIDAR_DTYPE)


class ModelCollectionRuntime:
    """复用同一 worker/世界逐 tick 推进模型闭环采集。"""

    def __init__(self, client, cfg, allocator, frame, lidar):
        self._client = client
        self._cfg = cfg
        self._cc = cfg.carla_collector
        self._allocator = allocator
        self._frame = frame
        self._lidar = lidar
        self._camera_names = tuple(cfg.data.driving.cameras)
        self._world = None
        self._tm = None
        self._ego = None
        self._rig = None
        self._crowd = None
        self._vehicle_ids = []
        self._navigator = None
        self._traffic_lights = []
        self._traffic_metadata = []
        self._pending = None
        self._cleanup_after_flush = False

    def start(self, map_name, seed, weather, route):
        """重载场景并返回第一份 10Hz 观测、世界真值和 2Hz 传感器帧。"""
        self._destroy_episode()
        random.seed(int(seed))
        np.random.seed(int(seed) % (2 ** 32))
        world, tm = session.load_scene_world(
            self._client, map_name, self._cc.simulation.fixed_delta_seconds,
            self._cc.traffic.tm_port, int(seed))
        self._world, self._tm = world, tm
        session.apply_weather(world, weather)
        try:
            self._ego = actors.spawn_ego_vehicle(
                world, self._cc.ego.vehicle_filter, route["start"])
            self._vehicle_ids = actors.spawn_traffic_vehicles(
                self._client, world, tm, self._cc.traffic.num_vehicles,
                self._cc.traffic.vehicle_filter)
            self._crowd = actors.spawn_walkers(
                self._client, world, self._cc.traffic.num_walkers,
                self._cc.traffic.walker_filter,
                self._cc.traffic.walker_running_pct,
                self._cc.traffic.walker_arrival_radius_m)
            try:
                self._navigator = RouteNavigator(
                    world.get_map(), route["start"], route["end"],
                    self._cfg.clone_loop.route)
            except RuntimeError:
                self._destroy_episode()
                return {"status": P.STATUS_UNREACHABLE}
            self._rig = SensorRig(
                world, self._ego, self._cc.cameras, self._cc.lidar)
        except Exception:
            self._destroy_episode()
            raise

        self._traffic_lights = sorted(
            world.get_actors().filter("*traffic_light*"), key=lambda item: item.id)
        self._traffic_metadata = traffic_control.traffic_light_metadata(
            self._traffic_lights)
        self._allocator.reset()
        self._pending = None
        self._cleanup_after_flush = False
        self._steps = 0
        self._captures = 0
        self._last_collision_events = 0
        self._collision_active = False
        self._collision_deadline = None
        self._collision_progress = 0.0
        self._collision_clear_steps = 0
        self._stuck_steps = 0
        self._distance = 0.0
        self._last_location = self._ego.get_location()

        brake = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
        frame_id = None
        for _ in range(self._cc.simulation.warmup_ticks):
            self._ego.apply_control(brake)
            frame_id = world.tick()
        if frame_id is None:
            self._ego.apply_control(brake)
            frame_id = world.tick()
        sample = self._rig.gather(
            frame_id, self._cc.worker.command_timeout_s)
        self._write_model_inputs(sample)
        navigation = self._navigator.observe(self._ego.get_transform())
        world_state = self._world_state(frame_id, navigation)
        sensor_frame = self._store_capture(frame_id, sample, world_state)
        self._captures += 1
        status = (P.STATUS_MAX_FRAMES
                  if self._captures >= self._cc.collection.max_frames_per_scene
                  else P.STATUS_RUNNING)
        result = {
            "status": status,
            "observation": self._observation(world_state, navigation),
            "world_state": world_state,
            "sensor_frame": sensor_frame,
            "pending_capture": False,
            "route_geometry": self._navigator.geometry,
            "static_meta": self._static_meta(),
        }
        if status != P.STATUS_RUNNING:
            self._destroy_episode()
        return result

    def step(self, control):
        """应用一次由 Winner 轨迹产生的控制并推进一个 10Hz 仿真步。"""
        if self._pending is not None:
            raise RuntimeError("存在待 flush 的传感器帧，不能继续推进模型闭环")
        self._crowd.retarget_arrived()
        self._ego.apply_control(carla.VehicleControl(
            throttle=float(control["throttle"]), steer=float(control["steer"]),
            brake=float(control["brake"])))
        frame_id = self._world.tick()
        self._steps += 1
        current = self._ego.get_location()
        self._distance += current.distance(self._last_location)
        self._last_location = current

        capture_due = self._steps % self._cc.collection.capture_every_n_ticks == 0
        keys = None if capture_due else self._model_sensor_keys()
        sample = self._rig.gather(
            frame_id, self._cc.worker.command_timeout_s, keys=keys)
        self._write_model_inputs(sample)
        navigation = self._navigator.observe(self._ego.get_transform())
        world_state = self._world_state(frame_id, navigation)
        status = self._status(world_state, navigation)
        sensor_frame = None
        pending = False
        if capture_due:
            try:
                sensor_frame = self._store_capture(frame_id, sample, world_state)
                self._captures += 1
            except ArenaFull:
                self._pending = (frame_id, sample, world_state)
                pending = True
            if (status == P.STATUS_RUNNING and not self._collision_active
                    and self._captures + (1 if pending else 0)
                    >= self._cc.collection.max_frames_per_scene):
                status = P.STATUS_MAX_FRAMES
        result = {
            "status": status,
            "observation": self._observation(world_state, navigation),
            "world_state": world_state,
            "sensor_frame": sensor_frame,
            "pending_capture": pending,
        }
        if status != P.STATUS_RUNNING:
            if pending:
                self._cleanup_after_flush = True
            else:
                self._destroy_episode()
        return result

    def flush_pending(self):
        """主进程落完上一段后 reset arena，并写入此前完整暂存的溢出帧。"""
        if self._pending is None:
            raise RuntimeError("没有待 flush 的模型传感器帧")
        self._allocator.reset()
        frame_id, sample, world_state = self._pending
        self._pending = None
        frame = self._store_capture(frame_id, sample, world_state)
        self._captures += 1
        if self._cleanup_after_flush:
            self._cleanup_after_flush = False
            self._destroy_episode()
        return {"sensor_frame": frame, "used_bytes": self._allocator.used}

    def _model_sensor_keys(self):
        keys = ["rgb/" + name for name in self._camera_names]
        if self._cc.lidar.enabled:
            keys.append("lidar")
        return keys

    def _write_model_inputs(self, sample):
        rgb = b"".join(_camera_bgr_bytes(sample["rgb/" + name])
                       for name in self._camera_names)
        self._frame.write(rgb)
        if not self._cc.lidar.enabled:
            self._lidar_count, self._lidar_valid = 0, False
            return
        raw = np.frombuffer(sample["lidar"].raw_data, dtype=_LIDAR_DTYPE)
        xyz = np.stack((raw["x"], raw["y"], raw["z"]), axis=1).astype(np.float32)
        if xyz.nbytes > self._lidar.size_bytes:
            raise RuntimeError("LiDAR 点数超过模型共享区容量，拒绝截断")
        self._lidar.write_prefix(np.ascontiguousarray(xyz).tobytes())
        self._lidar_count, self._lidar_valid = int(len(xyz)), True

    def _store_capture(self, frame_id, sample, world_state):
        blobs = collect.store_sensor_frame(self._allocator, sample)
        return {
            "frame_id": int(frame_id), "sim_time": world_state["sim_time"],
            "ego": world_state["ego"], "blobs": blobs,
            "bboxes": world_state["bboxes"],
            "traffic_light_states": world_state["traffic_light_states"],
            "relevant_traffic_control": world_state["relevant_traffic_control"],
        }

    def _world_state(self, frame_id, navigation):
        states = traffic_control.traffic_light_states(self._traffic_lights)
        ego = collect.ego_state(self._ego)
        return {
            "frame_id": int(frame_id),
            "sim_time": float(self._world.get_snapshot().timestamp.elapsed_seconds),
            "ego": ego,
            "ego_box": _local_bounding_box(self._ego.bounding_box),
            "bboxes": annotations.dynamic_bboxes(self._world, self._ego.id),
            "traffic_light_states": states,
            "relevant_traffic_control": traffic_control.relevant_traffic_control_route(
                self._navigator.geometry, navigation["route_progress_m"],
                self._traffic_metadata, states),
            "navigation": navigation,
            "speed_mps": _speed(self._ego),
            "speed_limit_mps": float(self._ego.get_speed_limit()) / 3.6,
            "collision_events": int(self._rig.collision_events),
            "lane_invasions": int(self._rig.lane_invasions),
            "step": int(self._steps),
        }

    def _observation(self, world_state, navigation):
        transform = self._ego.get_transform()
        velocity = self._ego.get_velocity()
        return {
            "pose": world_state["ego"]["transform"],
            "ego_box": world_state["ego_box"],
            "intrinsics": self._model_intrinsics(),
            "extrinsics": [self._rig.extrinsics[name] for name in self._camera_names],
            "lidar_count": self._lidar_count,
            "lidar_valid": self._lidar_valid,
            "target_point": navigation["target_point"],
            "ego_velocity": _world_vector_to_ego(
                velocity.x, velocity.y, transform.rotation.yaw),
            "speed_mps": world_state["speed_mps"],
            "route_deviation_m": navigation["route_deviation_m"],
            "route_completion": navigation["route_completion"],
            "end_distance_m": navigation["end_distance_m"],
            "distance_travelled_m": self._distance,
            "lane_invasions": world_state["lane_invasions"],
            "sim_time_s": world_state["sim_time"],
            "step": self._steps,
        }

    def _model_intrinsics(self):
        rig = {camera.name: camera for camera in self._cc.cameras.rig}
        width, height = self._cc.cameras.width, self._cc.cameras.height
        return [[width / (2.0 * math.tan(math.radians(rig[name].fov) * 0.5)),
                 width / (2.0 * math.tan(math.radians(rig[name].fov) * 0.5)),
                 width * 0.5, height * 0.5] for name in self._camera_names]

    def _static_meta(self):
        return {
            "static_bboxes": annotations.static_bboxes(self._world),
            "intrinsics": {
                camera.name: compute_intrinsics(
                    self._cc.cameras.width, self._cc.cameras.height, camera.fov)
                for camera in self._cc.cameras.rig},
            "extrinsics": self._rig.extrinsics,
            "lidar_extrinsic": self._rig.lidar_extrinsic,
            "traffic_lights": self._traffic_metadata,
        }

    def _status(self, state, navigation):
        cfg = self._cc.model_collection
        new_collision = state["collision_events"] > self._last_collision_events
        self._last_collision_events = state["collision_events"]
        if new_collision:
            if not self._collision_active:
                self._collision_active = True
                self._collision_deadline = self._steps + cfg.collision_followup_steps
                self._collision_progress = navigation["route_progress_m"]
            self._collision_clear_steps = 0
        elif self._collision_active:
            self._collision_clear_steps += 1

        if self._collision_active:
            recovered = (
                self._collision_clear_steps >= cfg.recovery_clear_steps
                and navigation["route_progress_m"] - self._collision_progress
                >= cfg.recovery_progress_m
                and navigation["route_deviation_m"]
                <= self._cfg.clone_loop.safety.max_route_deviation_m)
            if recovered:
                self._collision_active = False
                self._collision_deadline = None
            elif self._steps >= self._collision_deadline:
                return P.STATUS_COLLISION_UNRECOVERED

        if navigation["reached"] and not self._collision_active:
            return P.STATUS_SUCCESS
        if (not self._collision_active and navigation["route_deviation_m"]
                > self._cfg.clone_loop.safety.max_route_deviation_m):
            return P.STATUS_OFF_ROUTE
        if (not self._collision_active
                and self._captures >= self._cc.collection.max_frames_per_scene):
            return P.STATUS_MAX_FRAMES

        unjustified = (not self._collision_active
                       and state["speed_mps"] < cfg.stuck_speed_mps
                       and not self._has_stop_justification(state))
        self._stuck_steps = self._stuck_steps + 1 if unjustified else 0
        if not self._collision_active and self._stuck_steps >= cfg.stuck_steps:
            return P.STATUS_UNJUSTIFIED_STALL
        return P.STATUS_RUNNING

    def _has_stop_justification(self, state):
        cfg = self._cc.model_collection
        control = state["relevant_traffic_control"]
        if (control.get("valid") and control.get("state") == "red"
                and control.get("route_distance", float("inf"))
                <= cfg.stop_red_light_distance_m):
            return True
        ego_box = next((box for box in state["bboxes"]
                        if box.get("semantic") == "ego"), None)
        if ego_box is None:
            return False
        pose = state["ego"]["transform"]
        origin = np.asarray(pose[:2], dtype=np.float64)
        yaw = math.radians(float(pose[5]))
        forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
        for box in state["bboxes"]:
            if box.get("semantic") not in ("vehicle", "pedestrian"):
                continue
            delta = np.asarray(box["location"][:2], dtype=np.float64) - origin
            distance = float(np.linalg.norm(delta))
            if distance <= np.finfo(float).eps:
                angle = 0.0
            else:
                angle = math.degrees(math.acos(np.clip(
                    float(np.dot(delta / distance, forward)), -1.0, 1.0)))
            if angle <= cfg.stop_obstacle_half_angle_deg and _box_edge_distance(
                    ego_box, box) <= cfg.stop_obstacle_distance_m:
                return True
        return False

    def _destroy_episode(self):
        if self._rig is not None:
            self._rig.destroy()
        if self._world is not None:
            actors.destroy_scene_actors(
                self._client, self._world, self._ego, self._vehicle_ids,
                self._crowd.walker_ids if self._crowd is not None else [],
                self._crowd.controller_ids if self._crowd is not None else [])
        self._ego = None
        self._rig = None
        self._crowd = None
        self._vehicle_ids = []
        self._navigator = None
        self._pending = None
        self._cleanup_after_flush = False

    def close(self):
        """销毁模型场景 actor 并恢复异步模式。"""
        self._destroy_episode()
        if self._world is not None:
            settings = self._world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self._world.apply_settings(settings)
        if self._tm is not None:
            self._tm.set_synchronous_mode(False)


def _camera_bgr_bytes(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
        image.height, image.width, 4)
    return np.ascontiguousarray(array[:, :, :3]).tobytes()


def _local_bounding_box(box):
    return {
        "location": [box.location.x, box.location.y, box.location.z],
        "extent": [box.extent.x, box.extent.y, box.extent.z],
        "rotation": [box.rotation.roll, box.rotation.pitch, box.rotation.yaw],
    }


def _speed(vehicle):
    velocity = vehicle.get_velocity()
    return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)


def _world_vector_to_ego(x, y, yaw_deg):
    yaw = math.radians(yaw_deg)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return [cosine * x + sine * y, -sine * x + cosine * y]


def _box_corners(box):
    center = np.asarray(box["location"][:2], dtype=np.float64)
    extent = np.asarray(box["extent"][:2], dtype=np.float64)
    yaw = math.radians(float(box["rotation"][2]))
    rotation = np.array([[math.cos(yaw), -math.sin(yaw)],
                         [math.sin(yaw), math.cos(yaw)]], dtype=np.float64)
    local = np.array([[-extent[0], -extent[1]], [-extent[0], extent[1]],
                      [extent[0], extent[1]], [extent[0], -extent[1]]])
    return local.dot(rotation.T) + center


def _box_edge_distance(first, second):
    """返回两个平面 OBB 的真实边界净距；相交时为 0。"""
    a, b = _box_corners(first), _box_corners(second)
    axes = np.vstack((np.diff(a[[0, 1, 2, 3, 0]], axis=0),
                      np.diff(b[[0, 1, 2, 3, 0]], axis=0)))
    normals = np.stack((-axes[:, 1], axes[:, 0]), axis=1)
    separated = [max(float(np.min(a.dot(axis)) - np.max(b.dot(axis))),
                     float(np.min(b.dot(axis)) - np.max(a.dot(axis))))
                 for axis in normals]
    if max(separated) <= 0.0:
        return 0.0
    distances = [_segment_distance(a[index], a[(index + 1) % 4],
                                   b[other], b[(other + 1) % 4])
                 for index in range(4) for other in range(4)]
    return min(distances)


def _segment_distance(a0, a1, b0, b1):
    return min(_point_segment_distance(point, start, end)
               for point, start, end in ((a0, b0, b1), (a1, b0, b1),
                                         (b0, a0, a1), (b1, a0, a1)))


def _point_segment_distance(point, start, end):
    vector = end - start
    scale = float(np.dot(point - start, vector) /
                  max(np.dot(vector, vector), np.finfo(float).eps))
    projection = start + np.clip(scale, 0.0, 1.0) * vector
    return float(np.linalg.norm(point - projection))
