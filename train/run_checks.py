# 本文件为 train/run.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_runtime(model, dataset):
    """校验对象: main 运行前置 —— 模型可训练、数据集非空。"""
    if sum(1 for _ in model.trainable_parameters()) == 0:
        raise ValueError("模型无可训练参数，训练无意义。")
    if len(dataset) == 0:
        raise ValueError("数据集为空，检查 data.dataset.scene_root 与窗口配置。")
