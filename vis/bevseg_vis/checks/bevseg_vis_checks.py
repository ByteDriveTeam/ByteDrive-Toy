"""BEVSeg 可视化输入检查。"""

import numpy as np


def check_bevseg_history(history, semantic_layers):
    """检查对象: render_bevseg_history 的栅格历史与语义层。"""
    if len(history) != 1:
        raise ValueError("BEVSeg 单帧可视化必须提供一个样本")
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


def check_bevseg_reconstruction(history, reconstruction_logits, codes, semantic_layers,
                                semantic_threshold):
    """检查对象: render_bevseg_reconstruction 的重建张量、量化特征与阈值。"""
    check_bevseg_history(history, semantic_layers)
    expected = (len(history), len(semantic_layers) + 2, 256, 256)
    if not isinstance(reconstruction_logits, np.ndarray) \
            or reconstruction_logits.shape != expected:
        raise ValueError("BEVSeg 重建 logits 形状必须为 {}".format(expected))
    if not isinstance(codes, np.ndarray) or codes.ndim != 2 \
            or codes.shape[0] != 16 or codes.shape[1] < 3:
        raise ValueError("BEVSeg 压缩特征必须为 [16,D] 且 D>=3")
    if not np.isfinite(reconstruction_logits).all() or not np.isfinite(codes).all():
        raise ValueError("BEVSeg 重建 logits 与压缩特征必须全部有限")
    if not 0.0 < semantic_threshold < 1.0:
        raise ValueError("BEVSeg 语义显示阈值必须位于 (0,1)")

def check_privileged_bev_vae_prior_samples(
        history, prior_reconstruction_logits, semantic_layers, semantic_threshold):
    """校验对象: 特权 BEV VAE 先验采样的批量重建 logits。"""
    check_bevseg_history(history, semantic_layers)
    expected = (len(semantic_layers) + 2, 256, 256)
    if (not isinstance(prior_reconstruction_logits, np.ndarray) or
            prior_reconstruction_logits.ndim != 5 or
            prior_reconstruction_logits.shape[0] <= 0 or
            prior_reconstruction_logits.shape[1] != len(history) or
            tuple(prior_reconstruction_logits.shape[2:]) != expected):
        raise ValueError("特权 BEV VAE 先验采样 logits 形状错误: [S,T,C,256,256]")
    if not np.isfinite(prior_reconstruction_logits).all():
        raise ValueError("特权 BEV VAE 先验采样 logits 必须全部有限")
    if not 0.0 < semantic_threshold < 1.0:
        raise ValueError("语义显示阈值必须位于 (0,1)")


def check_privileged_bev_vae_reconstruction(
        history, reconstruction_logits, latent_mu, latent_logvar,
        semantic_layers, semantic_threshold):
    """校验对象: 特权 BEV VAE 渲染所需的重建 logits 与连续 latent。"""
    check_bevseg_history(history, semantic_layers)
    expected = (len(history), len(semantic_layers) + 2, 256, 256)
    if not isinstance(reconstruction_logits, np.ndarray) \
            or reconstruction_logits.shape != expected:
        raise ValueError("特权 BEV VAE 重建 logits 形状错误: {}".format(expected))
    for name, value in (("latent_mu", latent_mu), ("latent_logvar", latent_logvar)):
        if (not isinstance(value, np.ndarray) or value.ndim != 3 or
                value.shape[0] <= 0 or value.shape[1] <= 0 or
                value.shape[1] != value.shape[2]):
            raise ValueError("特权 BEV VAE {} 必须为 [D,H,H]".format(name))
        if not np.isfinite(value).all():
            raise ValueError("特权 BEV VAE {} 必须全部有限".format(name))
    if latent_mu.shape != latent_logvar.shape:
        raise ValueError("特权 BEV VAE mu/logvar 形状必须一致")
    if not np.isfinite(reconstruction_logits).all():
        raise ValueError("特权 BEV VAE 重建 logits 必须全部有限")
    if not 0.0 < semantic_threshold < 1.0:
        raise ValueError("语义显示阈值必须位于 (0,1)")
