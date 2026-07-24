def check_episode_override(value):
    """校验对象: run_closed_loop.max_episodes_override —— None 或非负整数。"""
    if value is not None and (not isinstance(value, int) or value < 0):
        raise ValueError("max_episodes_override 必须为非负整数或 None")


def check_output_root(output, repo_root):
    """校验对象: clone_loop.output.root —— 严格限制在项目目录内。"""
    try:
        output.relative_to(repo_root)
    except ValueError:
        raise ValueError("clone_loop.output.root 必须位于项目目录内: {}".format(output))


def check_routes(routes):
    """校验对象: worker 生成点经筛选后的路线队列 —— 至少应有一条可运行路线。"""
    if not routes:
        raise RuntimeError("闭环路线队列为空，请检查地图及 clone_loop.route 距离范围")


def check_runtime_versions(info):
    """校验对象: Py37 worker/CARLA 服务端版本 —— 闭环目标固定为 Python 3.7 与 CARLA 0.9.15。"""
    if info.get("python_version") != [3, 7]:
        raise RuntimeError("CARLA worker 必须运行于 Python 3.7，实际 {}".format(
            info.get("python_version")))
    if not str(info.get("carla_version", "")).startswith("0.9.15"):
        raise RuntimeError("闭环仅支持 CARLA 0.9.15，服务端实际 {}".format(
            info.get("carla_version")))
