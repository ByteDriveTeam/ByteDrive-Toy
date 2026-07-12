"""逐场景随机：种子与天气预设（决策与记录都在 collector 侧，便于复现）。

模块: collector/scenarios/scenarios.py
依赖: numpy
读取配置: 由调用方传入 weather.randomize 开关；预设清单由 worker 经 init 回传（属运行期数据，非配置）
对外接口:
    - random_seed(rng) -> int                              # 从主 rng 抽一个可记录的场景种子
    - random_weather(rng, enabled, presets) -> str | None  # 随机选一个 carla 内置天气预设名
说明: Design ④⑤。种子每场景随机但需记录——故由主 rng 抽取并随场景元数据落盘；碰撞重试时
      再抽新种子即可「换种子重试」。天气改用 carla 内置预设：可选预设名由 worker 枚举当前 CARLA
      版本得到并经 init 回传，collector 仅随机选取并记录预设名字符串，worker 侧据名解析后应用。
"""

import numpy as np


def random_seed(rng):
    """抽取 [0, 2^31) 内的场景种子。"""
    return int(rng.randint(0, 2 ** 31 - 1))


def random_weather(rng, enabled, presets):
    """随机选一个 carla 内置天气预设名；enabled=False 返回 None（worker 用地图默认天气）。"""
    if not enabled:
        return None
    return str(rng.choice(presets))
