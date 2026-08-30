from pathlib import Path


def check_dataset_root(root):
    """校验对象: WorldModelDataset 的 root —— 必须存在且至少含一个完整栅格场景。"""
    if not Path(root).is_dir():
        raise FileNotFoundError("世界模型栅格目录不存在：{}".format(root))


def check_scene_meta(meta, expected_shape, expected_layers):
    """校验对象: 栅格场景 meta —— 编码、shape 和图层契约必须与模型一致。"""
    if meta.get("format") != "bytedrive_bev_grid_v1" or not meta.get("complete"):
        raise ValueError("栅格场景格式不完整或不受支持")
    if tuple(meta.get("shape", ())) != tuple(expected_shape):
        raise ValueError("栅格 shape {} 与模型期望 {} 不一致".format(meta.get("shape"), expected_shape))
    if list(meta.get("layer_names", ())) != list(expected_layers):
        raise ValueError("栅格 layer_names 与 model.world_model.grid.layer_names 不一致")
