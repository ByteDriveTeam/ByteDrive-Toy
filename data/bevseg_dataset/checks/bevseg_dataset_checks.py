"""BEVSeg 数据集输入检查。"""

def check_bevseg_dataset(dataset):
    """检查对象: BevSegDataset，确保存在可用的完整单帧样本。"""
    if len(dataset) == 0:
        raise ValueError("BEVSeg 数据集为空，检查 scene_root、LMDB 和单帧历史")
