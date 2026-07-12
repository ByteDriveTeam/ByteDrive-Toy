"""场景读取器：合并单场景的 LMDB 与 mp4 为逐帧数据，探测各模态可用性。公开 API 重导出入口。

模块: vis/data_vis/reader/__init__.py
依赖: vis.data_vis.reader.reader
读取配置: —
对外接口:
    - list_scenes(root) -> list       # 枚举场景目录
    - SceneReader(scene_dir) -> 读取器  # 逐帧合并 LMDB 与 mp4
说明: 跨模块统一 `from vis.data_vis.reader import ...`；实现见 reader.py，入参校验见 checks/。
"""

from vis.data_vis.reader.reader import SceneReader, list_scenes

__all__ = ["SceneReader", "list_scenes"]
