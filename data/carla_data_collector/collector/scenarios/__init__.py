"""逐场景随机：种子与天气预设（决策与记录都在 collector 侧，便于复现）。公开 API 重导出入口。

模块: collector/scenarios/__init__.py
依赖: collector.scenarios.scenarios
读取配置: —（参数由调用方传入）
对外接口:
    - random_seed(rng)                    # 随机种子
    - random_weather(rng, enabled, presets)   # 随机天气预设
说明: 跨模块统一 `from collector import scenarios`（或 `from collector.scenarios import ...`）；实现见 scenarios.py（无校验）。
"""

from collector.scenarios.scenarios import random_seed, random_weather

__all__ = ["random_seed", "random_weather"]
