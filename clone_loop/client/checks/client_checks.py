def check_python_executable(path):
    """校验对象: WorkerClient.python_exe —— 配置的 Py37 解释器必须存在。"""
    if not path.is_file():
        raise FileNotFoundError("Py37 worker 解释器不存在: {}".format(path))


def check_response(response, command):
    """校验对象: WorkerClient._request 响应 —— 子进程必须返回成功协议消息。"""
    if response is None:
        raise RuntimeError("Py37 worker 在响应 {} 前退出".format(command))
    if not isinstance(response, dict) or "ok" not in response:
        raise RuntimeError("Py37 worker 对 {} 返回非法协议响应".format(command))
    if not response["ok"]:
        raise RuntimeError("Py37 worker 执行 {} 失败: {}".format(
            command, response.get("error")))
