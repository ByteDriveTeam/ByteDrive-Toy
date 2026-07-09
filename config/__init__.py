"""配置加载入口：读 yaml → 构造 schema → 校验 → 返回配置对象。

模块: config/__init__.py
依赖: pyyaml, config.schema
读取配置: —（提供加载能力，不绑定具体键）
对外接口:
    - load_config(path=None, env=None) -> Config   # 加载默认配置，可选用 <env>.yaml 覆盖
说明: 默认从同目录 default.yaml 加载；env 给定时叠加 <env>.yaml（仅覆盖出现的键）。
      这是全项目唯一的配置读取入口，实现文件应 from config import load_config。
"""

import copy
from pathlib import Path

from config.schema import build_config, validate_config

_CONFIG_DIR = Path(__file__).resolve().parent


def _deep_merge(base, override):
    """递归合并 override 到 base 之上（override 优先），返回新 dict，不改原值。"""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path):
    # 延迟导入 yaml：让 worker 侧（Py37，无 pyyaml）也能 import config.schema 而不被牵连
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(path=None, env=None):
    """加载配置对象。

    参数:
        path: 显式指定的主配置文件路径；缺省用 config/default.yaml
        env:  环境名；给定则叠加 config/<env>.yaml 覆盖默认值
    返回:
        校验通过的 Config 对象
    """
    base_path = Path(path) if path else _CONFIG_DIR / "default.yaml"
    raw = _read_yaml(base_path)

    if env:
        env_path = _CONFIG_DIR / "{}.yaml".format(env)
        raw = _deep_merge(raw, _read_yaml(env_path))

    cfg = build_config(raw)
    validate_config(cfg)
    return cfg
