# 本文件为 data/hd_map/hd_map.py 的校验伴随文件（规范 §7.1，免文件头）。

from pathlib import Path


def check_map_path(path):
    """校验对象: HdMap 构造入参 path —— HD 地图 npz 文件须存在。"""
    if not Path(path).is_file():
        raise FileNotFoundError("HD 地图文件不存在: {}（请将对应地图的 *_HD_map.npz 放入 data/map/）。".format(path))


def check_polylines(polylines, path):
    """校验对象: HdMap 解析结果 —— 至少解析出一条车道折线，否则地图为空/结构不符。"""
    if not polylines:
        raise ValueError("HD 地图 {} 未解析出任何车道折线（结构可能与预期不符）。".format(path))
