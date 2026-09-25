"""六层驾驶感知主干输入校验。"""


def check_transformer_inputs(image, rays, time, valid, dim, depths):
    """校验对象: DrivingTransformer.forward —— 内容、射线与时间须逐 Patch 对齐。"""
    if image.ndim != 3 or image.shape[-1] != dim:
        raise ValueError("图像内容须为 [B,N,D]")
    batch, tokens, _ = image.shape
    if tuple(rays.shape) != (batch, tokens, 5, depths, 3):
        raise ValueError("图像射线须为 [B,N,5,depths,3]")
    if tuple(time.shape) != (batch, tokens):
        raise ValueError("图像时间须为 [B,N]")
    if valid is not None and tuple(valid.shape) != (batch, tokens):
        raise ValueError("图像有效位须为 [B,N]")
