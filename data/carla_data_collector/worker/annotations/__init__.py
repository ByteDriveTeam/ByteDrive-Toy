"""带语义的包围框抽取：动态 actor（逐帧）与静态环境物体（每场景）。公开 API 重导出入口。

模块: worker/annotations/__init__.py
依赖: worker.annotations.annotations
读取配置: —
对外接口:
    - static_bboxes(world)          # 静态环境物体包围框（每场景一次）
    - dynamic_bboxes(world, ego_id) # 动态 actor 包围框（逐帧）
说明: 跨模块统一 `from worker.annotations import ...`；实现见 annotations.py（无校验）。
"""

from worker.annotations.annotations import dynamic_bboxes, static_bboxes

__all__ = ["static_bboxes", "dynamic_bboxes"]
