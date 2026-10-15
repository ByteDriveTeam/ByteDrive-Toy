"""渲染 BEVSeg 真值、压缩器重建结果与量化特征 PCA。

模块: vis/bevseg_vis/bevseg_vis.py
依赖: cv2, numpy, vis.bevseg_vis.checks.bevseg_vis_checks
读取配置: —（显示阈值由调用方显式传入）
对外接口:
    - render_bevseg_history(history, semantic_layers, frame_ids=None) -> ndarray
    - save_bevseg_history(history, semantic_layers, output_path, frame_ids=None) -> Path
    - render_bevseg_reconstruction(history, reconstruction_logits, codes, semantic_layers,
                                   semantic_threshold, frame_ids=None, metadata=None) -> ndarray
    - save_bevseg_reconstruction(history, reconstruction_logits, codes, semantic_layers,
                                 semantic_threshold, output_path, frame_ids=None,
                                 metadata=None) -> Path
说明: PCA 以 16 个量化空间 token 为样本、2048 维码字为特征，并固定映射 PC1/2/3 到 RGB。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from vis.bevseg_vis.checks.bevseg_vis_checks import (
    check_bevseg_history,
    check_bevseg_reconstruction,
    check_privileged_bev_vae_reconstruction,
    check_privileged_bev_vae_prior_samples,
)


__all__ = [
    "render_bevseg_history",
    "save_bevseg_history",
    "render_bevseg_reconstruction",
    "save_bevseg_reconstruction",
    "render_privileged_bev_vae_reconstruction",
    "save_privileged_bev_vae_reconstruction",
]


# BGR 颜色；语义顺序来自 config.data.bevseg.layers。
_COLORS = (
    (55, 180, 55),    # drivable
    (255, 220, 40),   # lane_centerline
    (40, 165, 255),   # lane_divider
    (180, 80, 220),   # road_boundary
    (255, 150, 40),   # pedestrian_crossing
    (40, 40, 240),    # vehicle
    (80, 180, 255),   # pedestrian
    (30, 30, 255),    # stop_line_red
    (40, 190, 255),   # stop_line_yellow
    (80, 220, 80),    # stop_line_green
)
_BACKGROUND = (22, 24, 28)
_PANEL_SIZE = 192


def _title_panel(image, title, color=(235, 235, 235)):
    panel = cv2.resize(image, (_PANEL_SIZE, _PANEL_SIZE), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(panel, (0, 0), (_PANEL_SIZE - 1, 22), (10, 12, 15), -1)
    cv2.putText(panel, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                color, 1, cv2.LINE_AA)
    return panel


def _ego_marker(canvas):
    # 该函数在 256px 原图和面板上都会被调用，中心必须根据当前画布尺寸计算。
    height, width = canvas.shape[:2]
    center = (width // 2, height // 2)
    cv2.drawMarker(canvas, center, (255, 255, 255), cv2.MARKER_CROSS, 14, 1)
    cx, cy = center
    scale = min(height, width) / 256.0
    tip = max(int(round(10 * scale)), 2)
    half = max(int(round(7 * scale)), 2)
    triangle = np.asarray([(cx, cy - tip), (cx - half, cy + half),
                           (cx + half, cy + half)], np.int32)
    cv2.fillConvexPoly(canvas, triangle, (0, 220, 255))


def _semantic_composite(semantic, layers, threshold=0.5):
    height, width = semantic.shape[-2:]
    canvas = np.full((height, width, 3), _BACKGROUND, np.uint8)
    # 可行驶底色低透明度叠加，避免遮住车道线和动态体。
    for index, color in enumerate(_COLORS[:len(layers)]):
        mask = semantic[index] > threshold
        if not np.any(mask):
            continue
        overlay = np.zeros_like(canvas)
        overlay[mask] = color
        alpha = 0.20 if layers[index] == "drivable" else 0.82
        canvas = cv2.addWeighted(canvas, 1.0, overlay, alpha, 0.0)
    return canvas


def _direction_overlay(canvas, direction):
    result = canvas.copy()
    norm = np.linalg.norm(direction, axis=0)
    valid = norm > 0.5
    rows, cols = np.nonzero(valid)
    if not len(rows):
        return result
    # 以有效车道像素为候选，再按 16px 网格去重；直接采样网格中心会错过细车道线。
    cells_w = (direction.shape[2] + 15) // 16
    cell_ids = (rows // 16) * cells_w + cols // 16
    _, selected = np.unique(cell_ids, return_index=True)
    rows, cols = rows[selected], cols[selected]
    for row, col in zip(rows, cols):
        right, front = direction[:, row, col] / norm[row, col]
        end = (int(round(col + right * 11)), int(round(row - front * 11)))
        cv2.arrowedLine(result, (int(col), int(row)), end, (245, 245, 245), 1,
                        cv2.LINE_AA, tipLength=0.25)
    return result


def _binary_panel(mask, title, color, threshold):
    image = np.full((*mask.shape, 3), _BACKGROUND, np.uint8)
    image[mask > threshold] = color
    _ego_marker(image)
    return _title_panel(image, title)


def _legend_panel(layers, width):
    """绘制颜色到语义类别的显式映射，避免只凭颜色猜测。"""
    height = 94
    panel = np.full((height, width, 3), (14, 16, 20), np.uint8)
    cv2.putText(panel, "Legend | triangle=ego/front | arrows=lane direction | PCA: PC1=R PC2=G PC3=B",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (235, 235, 235), 1, cv2.LINE_AA)
    columns = min(5, len(layers))
    cell_w = max(width // columns, 1)
    for index, name in enumerate(layers):
        row, column = divmod(index, columns)
        x = column * cell_w + 8
        y = 27 + row * 31
        color = tuple(int(value) for value in _COLORS[index])
        cv2.rectangle(panel, (x, y), (x + 14, y + 14), color, -1)
        cv2.putText(panel, name, (x + 20, y + 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (225, 225, 225), 1, cv2.LINE_AA)
    return panel


def _composite_panel(semantic, direction, layers, title, threshold):
    image = _semantic_composite(semantic, layers, threshold)
    image = _direction_overlay(image, direction)
    _ego_marker(image)
    return _title_panel(image, title)


def _detail_rows(semantic, direction, layers, tag, threshold):
    panels = [_binary_panel(semantic[index], "{} {}".format(tag, name),
                            _COLORS[index], threshold)
              for index, name in enumerate(layers)]
    direction_panel = _composite_panel(
        semantic, direction, layers, "{} current + direction".format(tag), threshold)
    blank = np.full_like(direction_panel, _BACKGROUND)
    return (
        np.hstack(panels[:5] + [direction_panel]),
        np.hstack(panels[5:] + [blank]),
    )


def _pca_rgb(codes):
    centered = codes.astype(np.float64, copy=False) - codes.mean(axis=0, keepdims=True)
    _, singular_values, axes = np.linalg.svd(centered, full_matrices=False)
    axes = axes[:3]
    # PCA 轴的正负号不唯一；以最大绝对载荷为正固定朝向，使同一特征的颜色可复现。
    anchors = np.argmax(np.abs(axes), axis=1)
    signs = np.where(axes[np.arange(3), anchors] < 0.0, -1.0, 1.0)
    projected = centered @ (axes * signs[:, None]).T
    tolerance = np.finfo(np.float64).eps * max(centered.shape) * singular_values[0]
    projected[:, singular_values[:3] <= tolerance] = 0.0
    low = projected.min(axis=0, keepdims=True)
    span = projected.max(axis=0, keepdims=True) - low
    normalized = np.divide(projected - low, span, out=np.full_like(projected, 0.5),
                           where=span > np.finfo(np.float64).eps)
    side = int(round(np.sqrt(codes.shape[0])))
    rgb = np.rint(normalized.reshape(side, side, 3) * 255.0).astype(np.uint8)
    return np.ascontiguousarray(rgb[..., ::-1])


def _pca_panel(codes):
    image = cv2.resize(_pca_rgb(codes), (_PANEL_SIZE, _PANEL_SIZE),
                       interpolation=cv2.INTER_NEAREST)
    step = _PANEL_SIZE // int(round(np.sqrt(codes.shape[0])))
    for coordinate in range(step, _PANEL_SIZE, step):
        cv2.line(image, (coordinate, 0), (coordinate, _PANEL_SIZE - 1), (35, 35, 35), 1)
        cv2.line(image, (0, coordinate), (_PANEL_SIZE - 1, coordinate), (35, 35, 35), 1)
    cv2.rectangle(image, (0, 0), (_PANEL_SIZE - 1, 22), (10, 12, 15), -1)
    cv2.putText(image, "quantized codes PCA (4x4)", (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
    return image


def _quality_metrics(target_semantic, pred_semantic, target_direction, pred_direction,
                     threshold):
    target_mask = target_semantic > 0.5
    pred_mask = pred_semantic > threshold
    intersection = np.logical_and(target_mask, pred_mask).sum(axis=(0, 2, 3))
    union = np.logical_or(target_mask, pred_mask).sum(axis=(0, 2, 3))
    mean_iou = (intersection[union > 0] / union[union > 0]).mean() if np.any(union) else 1.0

    target_norm = np.linalg.norm(target_direction, axis=1)
    pred_norm = np.linalg.norm(pred_direction, axis=1)
    valid = target_norm > 0.5
    cosine = (target_direction * pred_direction).sum(axis=1) / np.maximum(
        target_norm * pred_norm, np.finfo(np.float32).eps)
    direction_cosine = cosine[valid].mean() if np.any(valid) else None
    semantic_mae = np.abs(target_semantic - pred_semantic).mean()
    return (float(mean_iou),
            None if direction_cosine is None else float(direction_cosine),
            float(semantic_mae))


def _info_panel(metrics, codes, metadata):
    panel = np.full((_PANEL_SIZE, _PANEL_SIZE, 3), _BACKGROUND, np.uint8)
    direction_text = "{:.4f}".format(metrics[1]) if metrics[1] is not None else "n/a"
    lines = [
        "reconstruction summary",
        "semantic mIoU  {:.4f}".format(metrics[0]),
        "direction cos {}".format(direction_text),
        "semantic MAE   {:.4f}".format(metrics[2]),
        "codes          {}x{}".format(*codes.shape),
    ]
    lines.extend("{}  {}".format(key, value) for key, value in (metadata or {}).items())
    for index, line in enumerate(lines):
        color = (235, 235, 235) if index == 0 else (205, 210, 215)
        cv2.putText(panel, str(line), (7, 18 + index * 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, color, 1, cv2.LINE_AA)
    return panel


def render_bevseg_history(history, semantic_layers, frame_ids=None):
    """渲染单帧语义层和方向箭头，返回 BGR 画布。"""
    check_bevseg_history(history, semantic_layers)
    frame_ids = list(range(len(history))) if frame_ids is None else list(frame_ids)
    if len(frame_ids) != len(history):
        raise ValueError("frame_ids 数量必须与单帧样本一致")

    prepared = [(sample["semantic"], sample["direction"]) for sample in history]
    temporal = [_composite_panel(semantic, direction, semantic_layers,
                                 "frame {}".format(frame_id), 0.5)
                for (semantic, direction), frame_id in zip(prepared, frame_ids)]
    temporal_row = np.hstack(temporal)

    current_semantic, current_direction = prepared[-1]
    panels = [_binary_panel(current_semantic[index], name, _COLORS[index], 0.5)
              for index, name in enumerate(semantic_layers)]
    layer_rows = [np.hstack(panels[start:start + 5]) for start in (0, 5)]
    direction = _composite_panel(current_semantic, current_direction, semantic_layers,
                                 "current + lane direction", 0.5)
    bottom = np.hstack([layer_rows[0], direction])
    bottom2 = np.hstack([layer_rows[1], np.full_like(direction, _BACKGROUND)])
    width = max(temporal_row.shape[1], bottom.shape[1], bottom2.shape[1])

    def pad(row):
        if row.shape[1] == width:
            return row
        return cv2.copyMakeBorder(row, 0, 0, 0, width - row.shape[1],
                                  cv2.BORDER_CONSTANT, value=_BACKGROUND)

    canvas = np.vstack([pad(temporal_row), pad(bottom), pad(bottom2),
                        _legend_panel(semantic_layers, width)])
    cv2.putText(canvas, "BEVSeg | X=right, Y=front | ego=center", (8, canvas.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


def save_bevseg_history(history, semantic_layers, output_path, frame_ids=None):
    """渲染并保存 BEVSeg 画布。"""
    image = render_bevseg_history(history, semantic_layers, frame_ids)
    return _save_image(image, output_path)


def render_bevseg_reconstruction(history, reconstruction_logits, codes, semantic_layers,
                                 semantic_threshold, frame_ids=None, metadata=None):
    """渲染单帧真值/重建对照和量化压缩特征的三通道 PCA 图。"""
    check_bevseg_reconstruction(
        history, reconstruction_logits, codes, semantic_layers, semantic_threshold)
    frame_ids = list(range(len(history))) if frame_ids is None else list(frame_ids)
    if len(frame_ids) != len(history):
        raise ValueError("frame_ids 数量必须与 BEVSeg 历史帧一致")

    target_semantic = np.stack([sample["semantic"] for sample in history])
    target_direction = np.stack([sample["direction"] for sample in history])
    semantic_logits = reconstruction_logits[:, :len(semantic_layers)]
    pred_semantic = 1.0 / (1.0 + np.exp(-np.clip(semantic_logits, -30.0, 30.0)))
    pred_direction = reconstruction_logits[:, len(semantic_layers):]
    lane_mask = pred_semantic[:, 1:5].max(axis=1, keepdims=True) > semantic_threshold
    pred_direction = pred_direction * lane_mask

    target_panels = [_composite_panel(semantic, direction, semantic_layers,
                                      "target frame {}".format(frame_id), 0.5)
                     for semantic, direction, frame_id
                     in zip(target_semantic, target_direction, frame_ids)]
    pred_panels = [_composite_panel(semantic, direction, semantic_layers,
                                    "recon frame {}".format(frame_id), semantic_threshold)
                   for semantic, direction, frame_id
                   in zip(pred_semantic, pred_direction, frame_ids)]
    metrics = _quality_metrics(target_semantic, pred_semantic, target_direction,
                               pred_direction, semantic_threshold)
    target_details = _detail_rows(
        target_semantic[-1], target_direction[-1], semantic_layers, "target", 0.5)
    pred_details = _detail_rows(
        pred_semantic[-1], pred_direction[-1], semantic_layers, "recon", semantic_threshold)
    rows = [
        np.hstack(target_panels + [_info_panel(metrics, codes, metadata)]),
        *target_details,
        np.hstack(pred_panels + [_pca_panel(codes)]),
        *pred_details,
    ]
    width = max(row.shape[1] for row in rows)
    rows = [row if row.shape[1] == width else cv2.copyMakeBorder(
        row, 0, 0, 0, width - row.shape[1], cv2.BORDER_CONSTANT, value=_BACKGROUND)
            for row in rows]
    canvas = np.vstack(rows + [_legend_panel(semantic_layers, width)])
    cv2.putText(canvas, "BEVSeg compressor | X=right, Y=front | ego=center",
                (8, canvas.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


def save_bevseg_reconstruction(history, reconstruction_logits, codes, semantic_layers,
                               semantic_threshold, output_path, frame_ids=None, metadata=None):
    """保存 BEVSeg 压缩器单帧重建与 PCA 对照画布。"""
    image = render_bevseg_reconstruction(
        history, reconstruction_logits, codes, semantic_layers,
        semantic_threshold, frame_ids, metadata)
    return _save_image(image, output_path)


def _save_image(image, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise IOError("无法写入 BEVSeg 可视化: {}".format(output_path))
    return output_path


def _feature_pca_panel(features, title):
    """将空间 latent 的前三个 PCA 分量绘制为与 BEVSeg 相同尺寸的面板。"""
    features = np.asarray(features)
    if features.ndim == 3:
        channels, height, width = features.shape
        samples = features.transpose(1, 2, 0).reshape(height * width, channels)
    else:
        samples = features
        height = width = int(round(np.sqrt(samples.shape[0])))
    if samples.shape[1] < 3:
        samples = np.pad(samples, ((0, 0), (0, 3 - samples.shape[1])), mode="constant")
    image = cv2.resize(_pca_rgb(samples), (_PANEL_SIZE, _PANEL_SIZE),
                       interpolation=cv2.INTER_NEAREST)
    step = _PANEL_SIZE // max(height, 1)
    for coordinate in range(step, _PANEL_SIZE, step):
        cv2.line(image, (coordinate, 0), (coordinate, _PANEL_SIZE - 1), (35, 35, 35), 1)
        cv2.line(image, (0, coordinate), (_PANEL_SIZE - 1, coordinate), (35, 35, 35), 1)
    cv2.rectangle(image, (0, 0), (_PANEL_SIZE - 1, 22), (10, 12, 15), -1)
    cv2.putText(image, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (235, 235, 235), 1, cv2.LINE_AA)
    return image


def _vae_info_panel(metrics, latent_mu, latent_logvar, metadata):
    """绘制 VAE 重建指标、posterior 噪声和 latent 有效秩摘要。"""
    panel = np.full((_PANEL_SIZE, _PANEL_SIZE, 3), _BACKGROUND, np.uint8)
    flat = latent_mu.transpose(1, 2, 0).reshape(-1, latent_mu.shape[0]).astype(np.float64)
    centered = flat - flat.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(centered.shape[0] - 1, 1)
    eigenvalues = np.linalg.eigvalsh(covariance).clip(0)[::-1]
    effective_rank = ((eigenvalues.sum() ** 2) /
                      max((eigenvalues ** 2).sum(), np.finfo(np.float64).eps))
    posterior_std = np.exp(0.5 * np.clip(latent_logvar.astype(np.float64), -30.0, 20.0))
    kl = 0.5 * (latent_mu.astype(np.float64) ** 2 +
                np.exp(np.clip(latent_logvar.astype(np.float64), -30.0, 20.0)) -
                1.0 - latent_logvar.astype(np.float64)).mean()
    direction_text = "{:.4f}".format(metrics[1]) if metrics[1] is not None else "n/a"
    lines = [
        "VAE reconstruction summary",
        "semantic mIoU  {:.4f}".format(metrics[0]),
        "direction cos  {}".format(direction_text),
        "semantic MAE   {:.4f}".format(metrics[2]),
        "latent         {}x{}x{}".format(*latent_mu.shape),
        "effective rank {:.2f}".format(effective_rank),
        "posterior std  {:.4f}".format(posterior_std.mean()),
        "KL raw         {:.4f}".format(kl),
    ]
    lines.extend("{}  {}".format(key, value) for key, value in (metadata or {}).items())
    for index, line in enumerate(lines[:10]):
        color = (235, 235, 235) if index == 0 else (205, 210, 215)
        cv2.putText(panel, str(line), (7, 18 + index * 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.37, color, 1, cv2.LINE_AA)
    return panel


def _vae_prior_info_panel(metrics, sample_id, metadata):
    """绘制标准正态先验采样的重建指标。"""
    panel = np.full((_PANEL_SIZE, _PANEL_SIZE, 3), _BACKGROUND, np.uint8)
    direction_text = "{:.4f}".format(metrics[1]) if metrics[1] is not None else "n/a"
    lines = [
        "prior sample {}".format(sample_id),
        "semantic mIoU  {:.4f}".format(metrics[0]),
        "direction cos  {}".format(direction_text),
        "semantic MAE   {:.4f}".format(metrics[2]),
        "z ~ N(0,I)",
    ]
    lines.extend("{}  {}".format(key, value) for key, value in (metadata or {}).items())
    for index, line in enumerate(lines[:10]):
        color = (235, 235, 235) if index == 0 else (205, 210, 215)
        cv2.putText(panel, str(line), (7, 18 + index * 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.37, color, 1, cv2.LINE_AA)
    return panel


def render_privileged_bev_vae_reconstruction(
        history, reconstruction_logits, latent_mu, latent_logvar,
        semantic_layers, semantic_threshold, frame_ids=None, metadata=None,
        prior_reconstruction_logits=None, prior_sample_ids=None):
    """渲染特权 BEV VAE 的 target、重建、latent PCA 和逐层语义细节。"""
    check_privileged_bev_vae_reconstruction(
        history, reconstruction_logits, latent_mu, latent_logvar,
        semantic_layers, semantic_threshold)
    if prior_reconstruction_logits is not None:
        check_privileged_bev_vae_prior_samples(
            history, prior_reconstruction_logits, semantic_layers, semantic_threshold)
    frame_ids = list(range(len(history))) if frame_ids is None else list(frame_ids)
    if len(frame_ids) != len(history):
        raise ValueError("frame_ids 数量必须与特权 BEV VAE 历史帧一致")
    target_semantic = np.stack([sample["semantic"] for sample in history])
    target_direction = np.stack([sample["direction"] for sample in history])
    semantic_logits = reconstruction_logits[:, :len(semantic_layers)]
    pred_semantic = 1.0 / (1.0 + np.exp(-np.clip(semantic_logits, -30.0, 30.0)))
    pred_direction = reconstruction_logits[:, len(semantic_layers):]
    lane_mask = pred_semantic[:, 1:5].max(axis=1, keepdims=True) > semantic_threshold
    pred_direction = pred_direction * lane_mask
    target_panels = [_composite_panel(semantic, direction, semantic_layers,
                                      "target frame {}".format(frame_id), 0.5)
                     for semantic, direction, frame_id in
                     zip(target_semantic, target_direction, frame_ids)]
    pred_panels = [_composite_panel(semantic, direction, semantic_layers,
                                    "recon frame {}".format(frame_id), semantic_threshold)
                   for semantic, direction, frame_id in
                   zip(pred_semantic, pred_direction, frame_ids)]
    metrics = _quality_metrics(target_semantic, pred_semantic, target_direction,
                               pred_direction, semantic_threshold)
    target_details = _detail_rows(
        target_semantic[-1], target_direction[-1], semantic_layers, "target", 0.5)
    pred_details = _detail_rows(
        pred_semantic[-1], pred_direction[-1], semantic_layers, "recon", semantic_threshold)
    rows = [
        np.hstack(target_panels + [_vae_info_panel(metrics, latent_mu, latent_logvar, metadata)]),
        *target_details,
        np.hstack(pred_panels + [_feature_pca_panel(
            latent_mu, "latent mu PCA ({}x{})".format(latent_mu.shape[1], latent_mu.shape[2]))]),
        *pred_details,
    ]
    if prior_reconstruction_logits is not None:
        prior_sample_ids = (list(range(1, prior_reconstruction_logits.shape[0] + 1))
                            if prior_sample_ids is None else list(prior_sample_ids))
        if len(prior_sample_ids) != prior_reconstruction_logits.shape[0]:
            raise ValueError("prior_sample_ids 数量必须与先验采样数量一致")
        for sample_logits, sample_id in zip(prior_reconstruction_logits, prior_sample_ids):
            sample_semantic_logits = sample_logits[:, :len(semantic_layers)]
            sample_semantic = 1.0 / (1.0 + np.exp(-np.clip(sample_semantic_logits, -30.0, 30.0)))
            sample_direction = sample_logits[:, len(semantic_layers):]
            sample_lane_mask = sample_semantic[:, 1:5].max(axis=1, keepdims=True) > semantic_threshold
            sample_direction = sample_direction * sample_lane_mask
            sample_panels = [
                _composite_panel(semantic, direction, semantic_layers,
                                 "prior #{} frame {}".format(sample_id, frame_id),
                                 semantic_threshold)
                for semantic, direction, frame_id in
                zip(sample_semantic, sample_direction, frame_ids)]
            sample_metrics = _quality_metrics(
                target_semantic, sample_semantic, target_direction,
                sample_direction, semantic_threshold)
            rows.extend([
                np.hstack(sample_panels + [_vae_prior_info_panel(
                    sample_metrics, sample_id, metadata)]),
                *_detail_rows(sample_semantic[-1], sample_direction[-1],
                              semantic_layers, "prior #{}".format(sample_id),
                              semantic_threshold),
            ])
    width = max(row.shape[1] for row in rows)
    rows = [row if row.shape[1] == width else cv2.copyMakeBorder(
        row, 0, 0, 0, width - row.shape[1], cv2.BORDER_CONSTANT, value=_BACKGROUND)
            for row in rows]
    canvas = np.vstack(rows + [_legend_panel(semantic_layers, width)])
    prior_footer = " | prior z~N(0,I) samples={}".format(
        prior_reconstruction_logits.shape[0]) if prior_reconstruction_logits is not None else ""
    cv2.putText(canvas, "Privileged BEV VAE | X=right, Y=front | ego=center" + prior_footer,
                (8, canvas.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


def save_privileged_bev_vae_reconstruction(
        history, reconstruction_logits, latent_mu, latent_logvar,
        semantic_layers, semantic_threshold, output_path, frame_ids=None, metadata=None,
        prior_reconstruction_logits=None, prior_sample_ids=None):
    """保存特权 BEV VAE 的 target/重建/latent PCA 画布。"""
    image = render_privileged_bev_vae_reconstruction(
        history, reconstruction_logits, latent_mu, latent_logvar,
        semantic_layers, semantic_threshold, frame_ids, metadata,
        prior_reconstruction_logits, prior_sample_ids)
    return _save_image(image, output_path)
