"""把单相机的 BGR 帧序列编码为 H.265 mp4。

模块: collector/encode.py
依赖: av, numpy, collector.encode_checks
读取配置: 由 encode_camera 接收 output.video_codec/video_crf/video_fps 与相机分辨率，自身不读 config
对外接口:
    - encode_camera(frames_bgr, out_path, codec, crf, fps, width, height) -> int  # 返回写入帧数
说明: Design ⑧——仅 RGB 走 mp4，每相机每场景一个文件。frames_bgr 为产出 (H,W,3) uint8(BGR) 的可迭代对象
      （通常是从共享内存惰性读取的生成器，避免把整场景 RGB 同时驻留内存）。pix_fmt 用 yuv420p 保证通用可播放。
"""

import av
import numpy as np

from collector.encode_checks import check_frame


def encode_camera(frames_bgr, out_path, codec, crf, fps, width, height):
    """逐帧编码并落盘 mp4，返回实际写入帧数。"""
    container = av.open(str(out_path), mode="w")
    stream = container.add_stream(codec, rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": str(crf)}  # H.265 质量因子

    count = 0
    for bgr in frames_bgr:
        check_frame(bgr, height, width)
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(bgr), format="bgr24")
        container.mux(stream.encode(frame))  # 编码器内部按 pix_fmt 自动转换
        count += 1
    container.mux(stream.encode(None))  # flush 余下缓冲
    container.close()
    return count
