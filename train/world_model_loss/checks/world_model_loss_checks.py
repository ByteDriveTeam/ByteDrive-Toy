def check_reconstruction_outputs(outputs):
    """校验对象: WorldModelReconstructionLoss 输入 —— 四层预测/目标 shape 必须一致。"""
    predictions, targets = outputs["predictions"], outputs["targets"]
    if predictions.shape != targets.shape or predictions.ndim != 4 or predictions.shape[0] != 4:
        raise ValueError("预测/目标期望一致的 [4,B,N,D]，实际 {} / {}".format(
            tuple(predictions.shape), tuple(targets.shape)))
    if tuple(outputs["mask"].shape) != tuple(predictions.shape[1:3]):
        raise ValueError("mask 期望 [B,N]，实际 {}".format(tuple(outputs["mask"].shape)))
