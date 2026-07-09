"""Carla 世界/地图生命周期：连接、加载 Opt 地图、严格同步、天气与种子。

模块: worker/session.py
依赖: carla, worker.geometry, worker.session_checks
读取配置: 由 start_scene 传入标量（simulation.map / fixed_delta_seconds、traffic.tm_port 等），自身不读 config
对外接口:
    - connect(host, port, timeout_s) -> carla.Client
    - query_spawn_points(client, map_name) -> list[list[6]]   # 加载地图后取全部可达点
    - load_scene_world(client, map_name, fixed_delta_seconds, tm_port, seed) -> (world, tm)
    - list_weather_presets() -> list[str]                     # 当前 CARLA 版本内置的全部天气预设名
    - apply_weather(world, preset_name) -> None               # 按预设名设置天气；None 用地图默认
说明: 仅支持 Opt 地图，并 unload ParkedVehicles 图层以规避静态车辆已知 API 问题（Design ⑪）。
      严格同步模式 + 固定步长是多传感器对齐与可复现的前提（Design 同步要求）。
"""

import carla

from worker.geometry import transform_to_list
from worker.session_checks import check_map_name, check_weather_preset


def connect(host, port, timeout_s):
    """连接 Carla 服务端。"""
    client = carla.Client(host, port)
    client.set_timeout(timeout_s)
    return client


def _load_opt_map(client, map_name):
    """加载 Opt 地图并关闭静态车辆图层；返回 world。"""
    check_map_name(map_name)
    world = client.load_world(map_name)
    # 主动 unload 静态/停放车辆图层：该图层存在已知 API 问题，规避之（Design ⑪）
    world.unload_map_layer(carla.MapLayer.ParkedVehicles)
    return world


def query_spawn_points(client, map_name):
    """加载地图并返回全部推荐生成点（路线队列的可达点来源，Design ③）。"""
    world = _load_opt_map(client, map_name)
    return [transform_to_list(t) for t in world.get_map().get_spawn_points()]


def load_scene_world(client, map_name, fixed_delta_seconds, tm_port, seed):
    """为新场景重载地图并进入严格同步模式，返回 (world, traffic_manager)。

    每场景重载地图以彻底重置环境（Design ④）；TrafficManager 同步并按 seed 复现。
    """
    world = _load_opt_map(client, map_name)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = fixed_delta_seconds
    world.apply_settings(settings)

    tm = client.get_trafficmanager(tm_port)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(seed)  # 交通流随机可复现，seed 由 collector 记录
    return world, tm


def list_weather_presets():
    """枚举当前 CARLA 版本内置的全部天气预设名（collector 据此随机选取）。

    预设是 carla.WeatherParameters 的类属性且自身即 WeatherParameters 实例，
    故按此判定筛出，自动排除方法与非预设属性、不与 CARLA 版本写死的清单对不上。
    """
    return [name for name in dir(carla.WeatherParameters)
            if not name.startswith("_")
            and isinstance(getattr(carla.WeatherParameters, name), carla.WeatherParameters)]


def apply_weather(world, preset_name):
    """按 collector 下发的预设名设置世界天气（Design ⑤，随机决策在 collector 侧）。

    preset_name 为 None 时不改动，保留地图默认天气（randomize 关闭的情形）。
    """
    if preset_name is None:
        return
    check_weather_preset(preset_name)
    world.set_weather(getattr(carla.WeatherParameters, preset_name))
