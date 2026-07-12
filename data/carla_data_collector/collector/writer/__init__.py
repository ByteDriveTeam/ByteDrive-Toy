"""把场景的非 RGB 数据写入 LMDB（深度/语义Lidar/包围框/主车状态/元数据/视频引用）。公开 API 重导出入口。

模块: collector/writer/__init__.py
依赖: collector.writer.writer
读取配置: —（参数由调用方传入）
对外接口:
    - LmdbWriter(...)         # 场景 LMDB 写入器
    - read_scene_route(lmdb_path)   # 读回场景路线
    - pack_array / unpack_array     # 数组<->字节序列化
说明: 跨模块统一 `from collector.writer import ...`；实现见 writer.py（无校验）。
"""

from collector.writer.writer import LmdbWriter, pack_array, read_scene_route, unpack_array

__all__ = ["LmdbWriter", "read_scene_route", "pack_array", "unpack_array"]
