"""worker —— Py37 侧 Carla 采集子进程的实现包。

只在 py37_venv 下运行（Carla 0.9.15 仅兼容 Py3.7）。通过 stdin/stdout 的 JSON 行协议
受 Py312 collector 驱动，把每帧传感器数据写入共享内存 arena。
"""
