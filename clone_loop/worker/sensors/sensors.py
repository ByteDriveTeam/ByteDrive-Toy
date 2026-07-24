"""创建闭环前向 RGB 与安全事件传感器，并按仿真帧严格同步取图。

模块: clone_loop/worker/sensors/sensors.py
依赖: math, queue, numpy, carla
读取配置:
    clone_loop.camera.width / height / x / y / z / roll / pitch / yaw（由 cfg_camera 传入）
    model.driving.bev.fov_deg（由 fov_deg 构造参数传入）
对外接口:
    - ClosedLoopSensors(world, ego, cfg_camera, fov_deg)
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
    """闭环同步相机与安全事件传感器组。"""

    def __init__(self, world, ego, cfg_camera, fov_deg):
        self._actors = []
        self._queue = queue.Queue()
        self._collided = False
        self._lane_invasions = 0
        self._cfg = cfg_camera
        self._fov = fov_deg
        self._build(world, ego)

    def _build(self, world, ego):
        blueprints = world.get_blueprint_library()
        camera_bp = blueprints.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self._cfg.width))
        camera_bp.set_attribute("image_size_y", str(self._cfg.height))
        camera_bp.set_attribute("fov", str(self._fov))
        transform = carla.Transform(
            carla.Location(x=self._cfg.x, y=self._cfg.y, z=self._cfg.z),
            carla.Rotation(
                roll=self._cfg.roll, pitch=self._cfg.pitch, yaw=self._cfg.yaw))
        camera = world.spawn_actor(camera_bp, transform, attach_to=ego)
        camera.listen(self._queue.put)

        collision = world.spawn_actor(
            blueprints.find("sensor.other.collision"), carla.Transform(), attach_to=ego)
        collision.listen(self._on_collision)
        lane = world.spawn_actor(
            blueprints.find("sensor.other.lane_invasion"), carla.Transform(), attach_to=ego)
        lane.listen(self._on_lane_invasion)
        self._actors.extend((camera, collision, lane))

    def _on_collision(self, _event):
        self._collided = True

    def _on_lane_invasion(self, _event):
        self._lane_invasions += 1

    def gather(self, frame_id, timeout_s):
        """丢弃陈旧图像，直到取得与当前 world.tick 完全相同的 RGB 帧。"""
        while True:
            image = self._queue.get(timeout=timeout_s)
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
        """返回 `[fx,fy,cx,cy]`，与训练数据的模型输入顺序一致。"""
        focal = self._cfg.width / (
            2.0 * math.tan(math.radians(self._fov) * 0.5))
        return [focal, focal, self._cfg.width * 0.5, self._cfg.height * 0.5]

    @property
    def extrinsics(self):
        """返回相机相对 ego 的 `[x,y,z,roll,pitch,yaw]`。"""
        return [
            self._cfg.x, self._cfg.y, self._cfg.z,
            self._cfg.roll, self._cfg.pitch, self._cfg.yaw,
        ]

    def destroy(self):
        """停止并销毁本 episode 的全部传感器。"""
        for actor in self._actors:
            actor.stop()
            actor.destroy()
        self._actors = []
