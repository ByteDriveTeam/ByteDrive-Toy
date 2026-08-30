from pathlib import Path

import numpy as np


def check_grid_scene(scene_dir):
    """校验对象: BevGridReader 场景目录 —— 必须包含栅格 LMDB。"""
    if not (Path(scene_dir) / "lmdb" / "data.mdb").is_file():
        raise FileNotFoundError("栅格场景不存在或缺少 LMDB：{}".format(scene_dir))


def check_render_grid(grid, layer_names, colors):
    """校验对象: render_bev_grid 输入 —— 二值 [C,H,W] 与图层名/颜色数量一致。"""
    if not isinstance(grid, np.ndarray) or grid.ndim != 3:
        raise ValueError("可视化栅格期望 [C,H,W] ndarray")
    if len(grid) != len(layer_names) or len(grid) != len(colors):
        raise ValueError("栅格通道、图层名和颜色数量必须一致")
