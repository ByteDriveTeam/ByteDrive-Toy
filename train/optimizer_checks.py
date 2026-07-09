# 本文件为 train/optimizer.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_has_trainable(params):
    """校验对象: build_optimizer 的参数集合 —— 至少有一个可训练参数。"""
    if len(params) == 0:
        raise ValueError("模型无可训练参数（trunk/heads 是否被误冻结？）。")
