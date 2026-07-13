"""单帧场景数据集共享基类：索引/reader 缓存/RGB 归一化。公开 API 重导出入口。

模块: data/single_frame_base/__init__.py
依赖: data.single_frame_base.single_frame_base
读取配置: —
对外接口:
    - SingleFrameSceneBase(scene_root, camera, dino_mean, dino_std) -> Dataset
    - resolve_repo_path(path) -> Path
    - writable_contiguous(arr) -> np.ndarray
说明: 跨模块统一 `from data.single_frame_base import ...`；实现见 single_frame_base.py，校验见 checks/。
"""

from data.single_frame_base.single_frame_base import (
    SingleFrameSceneBase,
    resolve_repo_path,
    writable_contiguous,
)

__all__ = ["SingleFrameSceneBase", "resolve_repo_path", "writable_contiguous"]
