"""BEVSeg 可视化输入检查。"""


def check_bevseg_history(history, semantic_layers):
    """检查栅格器输出的五帧语义/方向形状。"""
    if len(history) != 5:
        raise ValueError("BEVSeg 可视化必须提供连续五帧")
    expected = len(semantic_layers)
    for sample in history:
        semantic = sample.get("semantic")
        direction = sample.get("direction")
        if semantic is None or direction is None:
            raise KeyError("BEVSeg 样本必须包含 semantic 和 direction")
        if semantic.shape != (expected, 256, 256):
            raise ValueError("BEVSeg semantic 形状必须为 [C,256,256]")
        if direction.shape != (2, 256, 256):
            raise ValueError("BEVSeg direction 形状必须为 [2,256,256]")
