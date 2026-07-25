"""创建闭环前向三目 RGB 与安全事件传感器，并按仿真帧严格同步取图。

模块: clone_loop/worker/sensors/sensors.py
依赖: math, queue, numpy, carla
读取配置:
    carla_collector.cameras.width / height / rig（由 cameras_cfg 传入）
    data.driving.cameras（由 camera_names 传入，定义相机轴顺序）
对外接口:
    - ClosedLoopSensors(world, ego, cameras_cfg, camera_names)
        .gather(frame_id, timeout_s) -> bytes
        .collided / .lane_invasions
        .intrinsics / .extrinsics
        .destroy() -> None
"""

import math
import queue

import carla
import numpy as np


__all__ = ["ClosedLoopSensors"]


class ClosedLoopSensors:
    """闭环同步三目相机与安全事件传感器组。"""

    def __init__(self, world, ego, cameras_cfg, camera_names):
        self._actors = []
        self._queues = {}
        self._collided = False
        self._lane_invasions = 0
        self._cfg = cameras_cfg
        rig = {camera.name: camera for camera in cameras_cfg.rig}
        self._cameras = [rig[name] for name in camera_names]
        self._build(world, ego)

    def _build(self, world, ego):
        blueprints = world.get_blueprint_library()
        camera_bp = blueprints.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self._cfg.width))
        camera_bp.set_attribute("image_size_y", str(self._cfg.height))
        for cfg in self._cameras:
            camera_bp.set_attribute("fov", str(cfg.fov))
            transform = carla.Transform(
                carla.Location(x=cfg.x, y=cfg.y, z=cfg.z),
                carla.Rotation(roll=cfg.roll, pitch=cfg.pitch, yaw=cfg.yaw))
            camera = world.spawn_actor(camera_bp, transform, attach_to=ego)
            sensor_queue = queue.Queue()
            camera.listen(sensor_queue.put)
            self._queues[cfg.name] = sensor_queue
            self._actors.append(camera)

        collision = world.spawn_actor(
            blueprints.find("sensor.other.collision"), carla.Transform(), attach_to=ego)
        collision.listen(self._on_collision)
        lane = world.spawn_actor(
            blueprints.find("sensor.other.lane_invasion"), carla.Transform(), attach_to=ego)
        lane.listen(self._on_lane_invasion)
        self._actors.extend((collision, lane))

    def _on_collision(self, _event):
        self._collided = True

    def _on_lane_invasion(self, _event):
        self._lane_invasions += 1

    def gather(self, frame_id, timeout_s):
        """严格收齐三路同帧 RGB，并按 data.driving.cameras 顺序拼接字节。"""
        return b"".join(self._gather_camera(
            self._queues[camera.name], frame_id, timeout_s) for camera in self._cameras)

    @staticmethod
    def _gather_camera(sensor_queue, frame_id, timeout_s):
        """丢弃单路陈旧图像，直到取得与当前 world.tick 完全相同的 RGB 帧。"""
        while True:
            image = sensor_queue.get(timeout=timeout_s)
            if image.frame == frame_id:
                array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
                    image.height, image.width, 4)
                return np.ascontiguousarray(array[:, :, :3]).tobytes()
            if image.frame > frame_id:
                raise RuntimeError(
                    "相机帧超前：sensor={} world={}".format(image.frame, frame_id))

    @property
    def collided(self):
        return self._collided

    @property
    def lane_invasions(self):
        return self._lane_invasions

    @property
    def intrinsics(self):
        """按相机轴顺序返回 `[V,4]` 内参。"""
        return [
            [
                self._cfg.width / (2.0 * math.tan(math.radians(cfg.fov) * 0.5)),
                self._cfg.width / (2.0 * math.tan(math.radians(cfg.fov) * 0.5)),
                self._cfg.width * 0.5, self._cfg.height * 0.5,
            ]
            for cfg in self._cameras
        ]

    @property
    def extrinsics(self):
        """按相机轴顺序返回相机相对 ego 的 `[V,6]` 外参。"""
        return [
            [cfg.x, cfg.y, cfg.z, cfg.roll, cfg.pitch, cfg.yaw]
            for cfg in self._cameras
        ]

    def destroy(self):
        """停止并销毁本 episode 的全部传感器。"""
        for actor in self._actors:
            actor.stop()
            actor.destroy()
        self._actors = []
        self._queues = {}
