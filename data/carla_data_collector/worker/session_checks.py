# 本文件为 worker/session.py 的校验伴随文件（规范 §7.1，免文件头）。

import carla


def check_map_name(map_name):
    """校验对象: session 加载的 map_name —— 运行期兜底确认仅用 Opt 地图。

    schema 已在加载期拦截（§7.3），此处仅防止 worker 收到异常下发值时误加载非 Opt 地图，
    因为非 Opt 地图无法 unload ParkedVehicles 图层，会触发 Design ⑪ 规避的已知问题。
    """
    assert isinstance(map_name, str) and map_name.endswith("_Opt"), \
        "worker 仅接受 Opt 地图，收到: {!r}".format(map_name)


def check_weather_preset(preset_name):
    """校验对象: apply_weather 收到的 preset_name —— 必须是 carla 内置天气预设名。

    预设名由 collector 从 worker 经 init 回传的清单中选取；此处兜底防止下发了
    当前 CARLA 版本不存在的名字，给出明确报错而非裸 AttributeError。
    """
    assert isinstance(getattr(carla.WeatherParameters, preset_name, None), carla.WeatherParameters), \
        "未知天气预设名: {!r}".format(preset_name)
