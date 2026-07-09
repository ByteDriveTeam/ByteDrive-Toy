"""ByteDrive 可视化包：只读消费 carla_data_collector 落盘的数据集并渲染。

模块: vis/data_vis/__init__.py
依赖: —
读取配置: —（各子模块经传入的 cfg.data_vis 读取，本文件不读 config）
对外接口: —（占位包标识；入口见 vis/data_vis/run.py）
说明: vis 层不依赖 carla，纯 numpy/opencv/av 重建投影与渲染，故可在 Py312 .venv 独立运行。
"""
