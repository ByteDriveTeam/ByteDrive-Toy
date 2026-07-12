# 本文件为 train/loop/loop.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_train_inputs(model, loader, optimizer):
    """校验对象: train_one_epoch 入参 —— 模型可训练、loader 与 optimizer 非空。"""
    if not hasattr(model, "trainable_parameters"):
        raise TypeError("model 需提供 trainable_parameters()（应为 PerceptionModel）。")
    if loader is None:
        raise ValueError("loader 不能为空。")
    if optimizer is None:
        raise ValueError("optimizer 不能为空。")
