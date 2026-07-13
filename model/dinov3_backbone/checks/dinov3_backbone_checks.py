# 本文件为 model/dinov3_backbone/dinov3_backbone.py 的校验伴随文件（规范 §7.1，免文件头）。


def check_backbone_cfg(cfg):
    """校验对象: DinoV3Backbone 构造入参 cfg —— 骨干关键字段合法。

    结构约束已在 config/schema 加载期拦截，此处仅挡「绕过 schema 直接构造」的裸 cfg。
    """
    if cfg.patch_size <= 0:
        raise ValueError("dinov3_backbone.patch_size 必须为正，实际为 {}。".format(cfg.patch_size))
    if cfg.hidden_dim <= 0:
        raise ValueError("dinov3_backbone.hidden_dim 必须为正，实际为 {}。".format(cfg.hidden_dim))
    if cfg.num_register_tokens < 0:
        raise ValueError(
            "dinov3_backbone.num_register_tokens 必须非负，实际为 {}。".format(cfg.num_register_tokens)
        )
    if len(cfg.feature_layers) == 0:
        raise ValueError("dinov3_backbone.feature_layers 至少需选一层。")


def check_feature_layers(num_hidden_states, feature_layers):
    """校验对象: cfg.feature_layers vs 骨干 hidden_states —— 每个层索引须落在可取范围内。"""
    if max(feature_layers) >= num_hidden_states:
        raise ValueError(
            "feature_layers 含索引 {} 超出 hidden_states 数量 {}（索引范围 0..{}）。".format(
                max(feature_layers), num_hidden_states, num_hidden_states - 1
            )
        )


def check_backbone_frames(frames, patch_size):
    """校验对象: DinoV3Backbone.forward 入参 frames —— 4 维 [N,3,H,W] 且 H/W 为 patch 整数倍。"""
    if frames.ndim != 4:
        raise ValueError("frames 期望 [N,3,H,W] 四维，实际 ndim={}。".format(frames.ndim))
    if frames.shape[1] != 3:
        raise ValueError("frames 通道数必须为 3（RGB），实际为 {}。".format(int(frames.shape[1])))
    height, width = int(frames.shape[2]), int(frames.shape[3])
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            "frames 高宽必须为 patch_size={} 的整数倍，实际为 {}×{}。".format(patch_size, height, width)
        )


def check_patch_tokens(sequence_length, expected_patches, num_register_tokens):
    """校验对象: 骨干输出序列长度 —— 至少含 1 CLS + register + 全部 patch，避免错切 token。"""
    minimum = 1 + num_register_tokens + expected_patches
    if sequence_length < minimum:
        raise ValueError(
            "骨干输出 token 数 {} 少于 1(CLS)+{}(register)+{}(patch)={}，无法取出 patch。".format(
                sequence_length, num_register_tokens, expected_patches, minimum
            )
        )
