# 本文件为 model/feature_fusion/feature_fusion.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_fusion_features(features, num_layers, hidden_dim):
    """校验对象: DinoFeatureFusion.forward 入参 features —— 4 维 [N,L,S,hidden] 且 L/hidden 匹配。"""
    if features.ndim != 4:
        raise ValueError("features 期望 [N,L,S,hidden] 四维，实际 ndim={}。".format(features.ndim))
    if int(features.shape[1]) != num_layers:
        raise ValueError(
            "features 层数 L 必须为 {}，实际为 {}。".format(num_layers, int(features.shape[1]))
        )
    if int(features.shape[-1]) != hidden_dim:
        raise ValueError(
            "features 单层末维必须为 {}，实际为 {}。".format(hidden_dim, int(features.shape[-1]))
        )
