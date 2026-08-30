"""读取离线栅格 LMDB，渲染彩色合成图与 10 个独立图层分块。

模块: vis/bev_grid_vis/bev_grid_vis.py
依赖: pathlib, time, cv2, lmdb, msgpack, numpy, config.schema.Config,
      data.world_model_dataset.decode_grid, 本模块 checks
读取配置:
    bev_grid_vis.root / scene / frame / play_fps / window_name / cell_px / display_scale /
        background_rgb / layer_colors / save_dir
对外接口:
    - BevGridReader(scene_dir) -> reader
    - render_bev_grid(grid, layer_names, cfg) -> ndarray
    - visualize_bev_grid(cfg, scene=None, frame=None, save=None, show=False) -> Path | None
说明: 合成图对重叠图层取颜色均值，不隐藏任何类别；分块固定按元数据 layer_names 标注。
      读取仅解压当前帧，顺序播放不预载场景，适合大 LMDB。
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import lmdb
import msgpack
import numpy as np

from config.schema import Config
from data.world_model_dataset import decode_grid
from vis.bev_grid_vis.checks.bev_grid_vis_checks import check_grid_scene, check_render_grid


__all__ = ["BevGridReader", "render_bev_grid", "visualize_bev_grid"]


class BevGridReader:
    """单场景栅格 LMDB 的按帧只读器。"""

    def __init__(self, scene_dir) -> None:
        self.scene_dir = Path(scene_dir)
        check_grid_scene(self.scene_dir)
        self._env = lmdb.open(str(self.scene_dir / "lmdb"), readonly=True, subdir=True, lock=False)
        with self._env.begin() as txn:
            self.meta = msgpack.unpackb(txn.get(b"meta"), raw=False)
        self.num_frames = int(self.meta["num_frames"])
        self.shape = tuple(self.meta["shape"])
        self.layer_names = tuple(self.meta["layer_names"])

    def frame(self, index):
        """解码指定帧为 `[10,H,W]` uint8。"""
        if index < 0 or index >= self.num_frames:
            raise IndexError("栅格帧索引越界：{} / {}".format(index, self.num_frames))
        with self._env.begin() as txn:
            blob = txn.get("grid/{:08d}".format(index).encode())
        return decode_grid(blob, self.shape)

    def close(self):
        """关闭 LMDB 句柄。"""
        self._env.close()


def render_bev_grid(grid, layer_names, cfg) -> np.ndarray:
    """生成包含彩色合成与逐层分块的 BGR 画布。"""
    vis = cfg.bev_grid_vis
    check_render_grid(grid, layer_names, vis.layer_colors)
    colors = np.asarray(vis.layer_colors, dtype=np.float32)
    masks = grid.astype(bool)
    count = masks.sum(0)
    mixed = np.einsum("chw,cd->hwd", masks.astype(np.float32), colors)
    background = np.asarray(vis.background_rgb[::-1], dtype=np.float32)
    composite = np.where(count[..., None] > 0, mixed / np.maximum(count[..., None], 1), background)
    panels = [_panel(composite.astype(np.uint8), "all_layers", vis.cell_px)]
    panels.extend(_panel(np.where(mask[..., None], color, background).astype(np.uint8), name, vis.cell_px)
                  for mask, color, name in zip(masks, colors, layer_names))
    blank = np.full_like(panels[0], background.astype(np.uint8))
    panels.extend([blank] * (12 - len(panels)))
    rows = [np.concatenate(panels[index:index + 4], axis=1) for index in range(0, 12, 4)]
    canvas = np.concatenate(rows, axis=0)
    if vis.display_scale != 1.0:
        canvas = cv2.resize(canvas, None, fx=vis.display_scale, fy=vis.display_scale,
                            interpolation=cv2.INTER_AREA)
    return canvas


def visualize_bev_grid(cfg: Config, scene=None, frame=None, save=None, show=False):
    """定位场景并保存单帧画布；show=True 时从起始帧顺序播放。"""
    vis = cfg.bev_grid_vis
    root = _repo_path(vis.root)
    scene_dir = _select_scene(root, scene if scene is not None else vis.scene)
    reader = BevGridReader(scene_dir)
    start = vis.frame if frame is None else frame
    output = _output_path(vis, reader.scene_dir.name, start, save)
    try:
        canvas = render_bev_grid(reader.frame(start), reader.layer_names, cfg)
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output), canvas):
                raise RuntimeError("写入栅格可视化失败：{}".format(output))
        if show:
            _play(reader, cfg, start)
        return output
    finally:
        reader.close()


def _play(reader, cfg, start):
    delay = 1.0 / cfg.bev_grid_vis.play_fps
    for index in range(start, reader.num_frames):
        begin = time.perf_counter()
        canvas = render_bev_grid(reader.frame(index), reader.layer_names, cfg)
        cv2.imshow(cfg.bev_grid_vis.window_name, canvas)
        elapsed_ms = int(max(delay - (time.perf_counter() - begin), 0.001) * 1000)
        if cv2.waitKey(elapsed_ms) & 0xFF in (27, ord("q")):
            break
    cv2.destroyWindow(cfg.bev_grid_vis.window_name)


def _panel(image, label, scale):
    panel = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(panel, (0, 0), (min(panel.shape[1] - 1, 360), 30), (0, 0, 0), -1)
    cv2.putText(panel, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                cv2.LINE_AA)
    return panel


def _select_scene(root, name):
    if name:
        path = root / name
        check_grid_scene(path)
        return path
    scenes = sorted(path for path in root.glob("scene_*") if path.is_dir())
    if not scenes:
        raise FileNotFoundError("没有可视化所需的栅格场景：{}".format(root))
    return scenes[0]


def _output_path(vis, scene_name, frame, explicit):
    if explicit is False:
        return None
    if explicit:
        return Path(explicit)
    return _repo_path(vis.save_dir) / "{}_{:06d}.png".format(scene_name, frame)


def _repo_path(value):
    path = Path(value)
    return path if path.is_absolute() else Path(__file__).resolve().parents[2] / path
