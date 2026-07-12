"""把单相机的 BGR 帧序列编码为 H.265 mp4。公开 API 重导出入口。

模块: collector/encode/__init__.py
依赖: collector.encode.encode
读取配置: —（编码参数由调用方传入）
对外接口:
    - encode_camera(frames_bgr, out_path, codec, crf, fps, width, height)   # 编码单相机 mp4
说明: 跨模块统一 `from collector.encode import ...`；实现见 encode.py，入参校验见 checks/。
"""

from collector.encode.encode import encode_camera

__all__ = ["encode_camera"]
