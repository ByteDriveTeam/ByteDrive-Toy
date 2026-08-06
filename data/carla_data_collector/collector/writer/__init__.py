"""把非 RGB 数据与独立运动学时间序列写入 LMDB。公开 API 重导出入口。

模块: collector/writer/__init__.py
依赖: collector.writer.writer
读取配置: —（参数由调用方传入）
对外接口:
    - LmdbWriter(...)         # 场景 LMDB 写入器
    - append_model_data(...)  # 回填模型 10Hz 真值、预测与代价
    - compact_lmdb(...)       # 逐键值校验后安全压实 LMDB
    - read_scene_route(lmdb_path)       # 读回场景路线
    - read_scene_map_route(lmdb_path)   # 读回场景地图与路线
    - read_scene_identity(lmdb_path)    # 读回地图、控制器与路线；忽略未完成模型行驶
    - pack_array / unpack_array     # 数组<->字节序列化
说明: 跨模块统一 `from collector.writer import ...`；实现见 writer.py（无校验）。
"""

from collector.writer.writer import (
    LmdbWriter,
    append_model_data,
    compact_lmdb,
    pack_array,
    read_scene_identity,
    read_scene_map_route,
    read_scene_route,
    unpack_array,
)

__all__ = [
    "LmdbWriter",
    "append_model_data",
    "compact_lmdb",
    "read_scene_identity",
    "read_scene_map_route",
    "read_scene_route",
    "pack_array",
    "unpack_array",
]
