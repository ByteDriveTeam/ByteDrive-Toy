"""BEVSeg 栅格合成的输入与输出检查。"""

def check_bevseg_sample(sample, semantic_layers):
    """检查对象: BEVSeg 样本的时间、通道和空间形状。"""
    semantic = sample["semantic"]
    direction = sample["direction"]
    if semantic.shape[1] != len(semantic_layers) or direction.shape[1] != 2:
        raise ValueError("BEVSeg 通道形状与配置不一致")
    if semantic.shape[-2:] != direction.shape[-2:]:
        raise ValueError("语义与方向空间尺寸不一致")
