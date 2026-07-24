"""管理 CARLA 世界、交通流、主车、路线和逐步闭环推进。

模块: clone_loop/worker/runtime/runtime.py
依赖: math, random, numpy, carla, clone_loop.protocol, clone_loop.worker.navigation,
      clone_loop.worker.sensors
读取配置:
    clone_loop.worker.carla_host / carla_port / startup_timeout_s / command_timeout_s
    clone_loop.simulation.map / fixed_delta_seconds / warmup_ticks / max_steps / random_weather
    clone_loop.route.* / traffic.num_vehicles / vehicle_filter / tm_port
    clone_loop.ego.vehicle_filter / camera.* / safety.max_route_deviation_m
    model.driving.bev.fov_deg
对外接口:
    - CarlaRuntime(cfg, shared_frame)
        .server_info() -> dict
        .query_spawn_points() -> list[list[6]]
        .reset(seed, route) -> dict
        .step(control) -> dict
        .close() -> None
"""

import math
import random

import carla
import numpy as np

from clone_loop import protocol as P
from clone_loop.worker.navigation import RouteNavigator
from clone_loop.worker.runtime.checks.runtime_checks import (
    check_blueprints,
    check_control,
    check_route,
)
from clone_loop.worker.sensors import ClosedLoopSensors


__all__ = ["CarlaRuntime"]

_SpawnActor = carla.command.SpawnActor
_SetAutopilot = carla.command.SetAutopilot
_FutureActor = carla.command.FutureActor
_DestroyActor = carla.command.DestroyActor


class CarlaRuntime:
    """单个 Py37 进程内串行复用的 CARLA 闭环运行时。"""

    def __init__(self, cfg, shared_frame):
        self._cfg = cfg.clone_loop
        self._camera_fov = cfg.model.driving.bev.fov_deg
        self._frame = shared_frame
        self._client = carla.Client(
            self._cfg.worker.carla_host, self._cfg.worker.carla_port)
        self._client.set_timeout(self._cfg.worker.startup_timeout_s)
        self._world = None
        self._tm = None
        self._ego = None
        self._sensors = None
        self._vehicle_ids = []
        self._navigator = None

    def server_info(self):
        """返回客户端实际连接到的 CARLA 服务端版本。"""
        version = self._client.get_server_version()
        self._client.set_timeout(self._cfg.worker.command_timeout_s)
        return {"carla_version": version}

    def query_spawn_points(self):
        """加载配置地图并返回 CARLA 推荐生成点。"""
        world = self._load_world()
        return [_transform_list(item) for item in world.get_map().get_spawn_points()]

    def reset(self, seed, route):
        """销毁旧 episode，重载地图并生成首帧闭环观测。"""
        check_route(route)
        self._destroy_episode()
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        self._world = self._load_world()
        self._configure_sync(seed)
        self._apply_weather(seed)
        self._ego = self._spawn_ego(route["start"])
        self._vehicle_ids = self._spawn_traffic()
        self._sensors = ClosedLoopSensors(
            self._world, self._ego, self._cfg.camera, self._camera_fov)
        self._navigator = RouteNavigator(
            self._world.get_map(), route["start"], route["end"], self._cfg.route)
        self._steps = 0
        self._distance = 0.0
        self._last_location = self._ego.get_location()

        brake = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
        frame_id = None
        for _ in range(self._cfg.simulation.warmup_ticks):
            self._ego.apply_control(brake)
            frame_id = self._world.tick()
        if frame_id is None:
            self._ego.apply_control(brake)
            frame_id = self._world.tick()
        self._frame.write(self._sensors.gather(
            frame_id, self._cfg.worker.command_timeout_s))
        navigation = self._navigator.observe(self._ego.get_transform())
        return self._observation(self._status(navigation), navigation)

    def step(self, control):
        """应用一次模型控制并推进一个同步仿真步，返回下一观测。"""
        check_control(control)
        self._ego.apply_control(carla.VehicleControl(
            throttle=float(control["throttle"]),
            steer=float(control["steer"]),
            brake=float(control["brake"])))
        frame_id = self._world.tick()
        self._frame.write(self._sensors.gather(
            frame_id, self._cfg.worker.command_timeout_s))
        self._steps += 1
        current = self._ego.get_location()
        self._distance += current.distance(self._last_location)
        self._last_location = current
        navigation = self._navigator.observe(self._ego.get_transform())
        return self._observation(self._status(navigation), navigation)

    def _status(self, navigation):
        if self._sensors.collided:
            return P.STATUS_COLLISION
        if navigation["reached"]:
            return P.STATUS_SUCCESS
        if navigation["route_deviation_m"] > self._cfg.safety.max_route_deviation_m:
            return P.STATUS_OFF_ROUTE
        if self._steps >= self._cfg.simulation.max_steps:
            return P.STATUS_MAX_STEPS
        return P.STATUS_RUNNING

    def _observation(self, status, navigation=None):
        transform = self._ego.get_transform()
        velocity = self._ego.get_velocity()
        navigation = navigation or self._navigator.observe(transform)
        ego_velocity = _world_vector_to_ego(
            velocity.x, velocity.y, transform.rotation.yaw)
        return {
            "status": status,
            "pose": _transform_list(transform),
            "intrinsics": self._sensors.intrinsics,
            "extrinsics": self._sensors.extrinsics,
            "target_point": navigation["target_point"],
            "ego_velocity": ego_velocity,
            "speed_mps": _speed(self._ego),
            "route_deviation_m": navigation["route_deviation_m"],
            "route_completion": navigation["route_completion"],
            "end_distance_m": navigation["end_distance_m"],
            "distance_travelled_m": self._distance,
            "lane_invasions": self._sensors.lane_invasions,
            "sim_time_s": self._world.get_snapshot().timestamp.elapsed_seconds,
            "step": self._steps,
        }

    def _load_world(self):
        world = self._client.load_world(self._cfg.simulation.map)
        world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        return world

    def _configure_sync(self, seed):
        settings = self._world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self._cfg.simulation.fixed_delta_seconds
        self._world.apply_settings(settings)
        self._tm = self._client.get_trafficmanager(self._cfg.traffic.tm_port)
        self._tm.set_synchronous_mode(True)
        self._tm.set_random_device_seed(seed)

    def _apply_weather(self, seed):
        if not self._cfg.simulation.random_weather:
            return
        presets = [
            getattr(carla.WeatherParameters, name)
            for name in dir(carla.WeatherParameters)
            if not name.startswith("_")
            and isinstance(getattr(carla.WeatherParameters, name), carla.WeatherParameters)
        ]
        self._world.set_weather(random.Random(seed).choice(presets))

    def _spawn_ego(self, pose):
        blueprints = self._world.get_blueprint_library().filter(
            self._cfg.ego.vehicle_filter)
        check_blueprints(blueprints, self._cfg.ego.vehicle_filter)
        blueprint = blueprints[0]
        blueprint.set_attribute("role_name", "hero")
        ego = self._world.spawn_actor(blueprint, _make_transform(pose))
        self._world.tick()
        return ego

    def _spawn_traffic(self):
        blueprints = self._world.get_blueprint_library().filter(
            self._cfg.traffic.vehicle_filter)
        check_blueprints(blueprints, self._cfg.traffic.vehicle_filter)
        spawn_points = self._world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        def command(transform):
            blueprint = random.choice(blueprints)
            blueprint.set_attribute("role_name", "autopilot")
            return _SpawnActor(blueprint, transform).then(
                _SetAutopilot(_FutureActor, True, self._tm.get_port()))

        results = self._client.apply_batch_sync([
            command(transform)
            for transform in spawn_points[:self._cfg.traffic.num_vehicles]
        ], True)
        return [result.actor_id for result in results if not result.error]

    def _destroy_episode(self):
        if self._sensors is not None:
            self._sensors.destroy()
        ids = self._vehicle_ids + ([self._ego.id] if self._ego is not None else [])
        if ids:
            self._client.apply_batch([_DestroyActor(actor_id) for actor_id in ids])
        self._sensors = None
        self._vehicle_ids = []
        self._ego = None
        self._navigator = None

    def close(self):
        """销毁 actor 并恢复异步模式，避免服务端停留在同步等待态。"""
        self._destroy_episode()
        if self._world is not None:
            settings = self._world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self._world.apply_settings(settings)
        if self._tm is not None:
            self._tm.set_synchronous_mode(False)


def _make_transform(pose):
    return carla.Transform(
        carla.Location(x=pose[0], y=pose[1], z=pose[2]),
        carla.Rotation(roll=pose[3], pitch=pose[4], yaw=pose[5]))


def _transform_list(transform):
    location, rotation = transform.location, transform.rotation
    return [
        location.x, location.y, location.z,
        rotation.roll, rotation.pitch, rotation.yaw,
    ]


def _speed(vehicle):
    velocity = vehicle.get_velocity()
    return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)


def _world_vector_to_ego(x, y, yaw_deg):
    yaw = math.radians(yaw_deg)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return [cosine * x + sine * y, -sine * x + cosine * y]
