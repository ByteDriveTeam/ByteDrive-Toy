"""驾驶训练样本监督面板的公开渲染接口。

模块: vis/data_vis/driving_sample/__init__.py
依赖: vis.data_vis.driving_sample.driving_sample
读取配置: —
对外接口:
    - render_driving_sample(sample, cfg, scene_name, frame_idx) -> np.ndarray
"""

from vis.data_vis.driving_sample.driving_sample import render_driving_sample

__all__ = ["render_driving_sample"]
