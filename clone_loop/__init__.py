"""CARLA 0.9.15 行为克隆闭环包：连接异构仿真 worker 与主环境模型推理。

模块: clone_loop/__init__.py
依赖: clone_loop.orchestrator（调用时延迟导入，保证 Py37 worker 不加载 torch）
读取配置: —（由公开入口转交各模块读取）
对外接口:
    - run_closed_loop(cfg, max_episodes_override=None) -> dict
"""


def run_closed_loop(cfg, max_episodes_override=None):
    """延迟导入主环境编排器，避免 Py37 仅导入协议时触发 PyTorch 依赖。"""
    from clone_loop.orchestrator import run_closed_loop as _run
    return _run(cfg, max_episodes_override)


__all__ = ["run_closed_loop"]
