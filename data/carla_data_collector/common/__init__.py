"""common —— Py37 worker 与 Py312 collector 两端共享的纯 Python 层。

仅含控制管道协议(protocol)与共享内存(shm)，不依赖 carla、不依赖重型第三方库，
以保证在 3.7 与 3.12 两个解释器下均可导入。
"""
