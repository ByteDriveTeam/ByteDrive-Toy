"""闭环驾驶与逐帧推理录像器的公开 API 重导出入口。

模块: clone_loop/recorder/__init__.py
依赖: clone_loop.recorder.recorder
读取配置: —
对外接口:
    - EpisodeRecorder(run_dir, episode_index, cfg)
"""

from clone_loop.recorder.recorder import EpisodeRecorder

__all__ = ["EpisodeRecorder"]
