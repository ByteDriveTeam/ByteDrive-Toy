# 本文件为 collector/worker_proc/worker_proc.py 的校验伴随文件（规范 §7.1，免文件头）。

import os


def check_python_exe(python_exe):
    """校验对象: WorkerProcess 入参 python_exe —— 必须指向真实存在的解释器。

    py37 解释器路径错误会让子进程静默拉起失败，提前在此明确报错。
    """
    assert os.path.isfile(str(python_exe)), "worker python_exe 不存在: {!r}".format(python_exe)
