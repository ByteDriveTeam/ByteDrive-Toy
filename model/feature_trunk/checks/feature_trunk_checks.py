# 本文件为 model/feature_trunk/feature_trunk.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_trunk_features(features, channels, grid_height, grid_width):
    """校验对象: FeatureTrunk.forward 入参 —— [B,S,C] 且序列尾部可容纳完整 patch 网格。"""
    if features.ndim != 3:
        raise ValueError("features 期望 [B,S,C] 三维，实际 ndim={}。".format(features.ndim))
    if int(features.shape[-1]) != channels:
        raise ValueError(
            "features 末维必须为工作维 {}，实际为 {}。".format(channels, int(features.shape[-1]))
        )
    if grid_height < 1 or grid_width < 1:
        raise ValueError("grid_height/grid_width 必须为正，实际 {} × {}。".format(grid_height, grid_width))
    patch_count = grid_height * grid_width
    if int(features.shape[1]) <= patch_count:
        raise ValueError(
            "features 须在 {} 个 patch 前保留 CLS/寄存器前缀，实际序列长 {}。".format(
                patch_count, int(features.shape[1])
            )
        )
