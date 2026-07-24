"""逐 episode 编码前向驾驶实况，并合成模型全部在线推理输出的诊断录像。

模块: clone_loop/recorder/recorder.py
依赖: fractions, pathlib, av, cv2, numpy, data.driving_targets, vis.driving_vis.render,
      clone_loop.recorder.checks.recorder_checks
读取配置:
    clone_loop.recording.enabled / codec / crf / tile_size_px
    clone_loop.camera.width / height
    clone_loop.simulation.fixed_delta_seconds
    model.driving.bev.* / model.driving.fields.up_channels
    driving_vis.field_colormap / lane_map.* / traffic_control.*
对外接口:
    - EpisodeRecorder(run_dir, episode_index, cfg)
        .write(frame_bgr, decision, observation, command) -> None
        .write_terminal(frame_bgr) -> None
        .artifacts -> dict
        .close() -> None
说明: 驾驶录像保存原始前向 RGB；推理录像为三行画布：相机/HUD、三场、道路线/交通控制/多模态轨迹。
"""

from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

from clone_loop.recorder.checks.recorder_checks import check_video_frame
from data.driving_targets import BevParams, inview_mask
from vis.driving_vis import render


__all__ = ["EpisodeRecorder"]

_BACKGROUND = (24, 24, 26)
_DRIVING_NAME = "episode_{:04d}_driving.mp4"
_INFERENCE_NAME = "episode_{:04d}_inference.mp4"


class _VideoSink:
    """固定尺寸 BGR 帧的增量 PyAV 编码器。"""

    def __init__(self, path, codec, crf, fps, width, height):
        self._width = width
        self._height = height
        self._container = av.open(str(path), mode="w")
        self._stream = self._container.add_stream(codec, rate=fps)
        self._stream.width = width
        self._stream.height = height
        self._stream.pix_fmt = "yuv420p"
        self._stream.options = {"crf": str(crf)}
        self._closed = False

    def write(self, frame_bgr):
        check_video_frame(frame_bgr, self._height, self._width)
        frame = av.VideoFrame.from_ndarray(
            np.ascontiguousarray(frame_bgr), format="bgr24")
        self._container.mux(self._stream.encode(frame))

    def close(self):
        if self._closed:
            return
        self._container.mux(self._stream.encode(None))
        self._container.close()
        self._closed = True


class EpisodeRecorder:
    """一条闭环路线的驾驶实况与模型推理双路录像器。"""

    def __init__(self, run_dir, episode_index, cfg):
        self._cfg = cfg
        self._recording = cfg.clone_loop.recording
        self._enabled = self._recording.enabled
        self._driving = None
        self._inference = None
        self._artifacts = {}
        if not self._enabled:
            return
        self._prepare_geometry()
        fps = Fraction(1, 1) / Fraction(
            str(cfg.clone_loop.simulation.fixed_delta_seconds))
        driving_name = _DRIVING_NAME.format(episode_index)
        inference_name = _INFERENCE_NAME.format(episode_index)
        self._artifacts = {
            "driving_video": driving_name,
            "inference_video": inference_name,
        }
        try:
            self._driving = _VideoSink(
                Path(run_dir) / driving_name, self._recording.codec, self._recording.crf,
                fps, cfg.clone_loop.camera.width, cfg.clone_loop.camera.height)
            canvas_size = self._recording.tile_size_px * 3
            self._inference = _VideoSink(
                Path(run_dir) / inference_name, self._recording.codec, self._recording.crf,
                fps, canvas_size, canvas_size)
        except Exception:
            self.close()
            raise

    def _prepare_geometry(self):
        bev_cfg = self._cfg.model.driving.bev
        scale = 2 ** len(self._cfg.model.driving.fields.up_channels)
        self._bev = BevParams(
            bev_cfg.x_min_m, bev_cfg.x_max_m, bev_cfg.y_min_m, bev_cfg.y_max_m,
            bev_cfg.height * scale, bev_cfg.width * scale)
        self._inview = inview_mask(self._bev, bev_cfg.fov_deg)

    def write(self, frame_bgr, decision, observation, command):
        """写入当前模型输入画面和与之对应的全部推理输出画布。"""
        if not self._enabled:
            return
        self._driving.write(frame_bgr)
        self._inference.write(
            self._render_inference(frame_bgr, decision, observation, command))

    def write_terminal(self, frame_bgr):
        """终态没有下一次模型推理，仅把最后一帧追加到驾驶实况。"""
        if self._enabled:
            self._driving.write(frame_bgr)

    @property
    def artifacts(self):
        """返回相对运行目录的双路录像文件名。"""
        return dict(self._artifacts)

    def close(self):
        """刷新编码器缓冲并关闭 MP4 容器。"""
        inference, driving = self._inference, self._driving
        self._inference = None
        self._driving = None
        try:
            if inference is not None:
                inference.close()
        finally:
            if driving is not None:
                driving.close()

    def _render_inference(self, frame_bgr, decision, observation, command):
        visual = decision["visualization"]
        tile = self._recording.tile_size_px
        field_colormap = self._cfg.driving_vis.field_colormap
        risk = render.colorize_field(visual["risk"], field_colormap, self._inview)
        drivable = render.colorize_field(
            visual["drivable"], field_colormap, self._inview)
        distribution = render.colorize_field(
            visual["distribution"], field_colormap, self._inview)
        lane = self._lane_panel(visual)
        traffic = self._traffic_panel(visual)
        trajectory = self._trajectory_panel(visual, traffic, decision)

        camera = _camera_panel(frame_bgr, tile * 3, tile, decision, observation, command)
        fields = _panel_row(
            (risk, drivable, distribution), ("risk", "drivable", "distribution"), tile)
        structure = _panel_row(
            (lane, traffic, trajectory), ("lanes", "traffic", "trajectories"), tile)
        return np.ascontiguousarray(np.vstack((camera, fields, structure)))

    def _lane_panel(self, visual):
        lane_cfg = self._cfg.driving_vis.lane_map
        return render.colorize_lane_map(
            visual["lane_class"], visual["lane_direction"],
            lane_cfg.class_colors, lane_cfg.arrow_color, lane_cfg.arrow_stride_px,
            lane_cfg.arrow_length_px, lane_cfg.arrow_thickness,
            lane_cfg.arrow_tip_ratio, self._inview)

    def _traffic_panel(self, visual):
        traffic_cfg = self._cfg.driving_vis.traffic_control
        stop = np.where(
            visual["stop_line"] > traffic_cfg.line_threshold,
            visual["stop_line"], 0.0)
        return render.colorize_traffic_control(
            stop, visual["traffic_state"], None, traffic_cfg.state_colors,
            traffic_cfg.unknown_color, self._inview)

    def _trajectory_panel(self, visual, traffic, decision):
        traffic_cfg = self._cfg.driving_vis.traffic_control
        base = render.bev_scene_composite(
            visual["risk"], visual["drivable"], visual["distribution"], self._inview)
        stop_mask = visual["stop_line"] > traffic_cfg.line_threshold
        base = render.overlay_traffic_control(
            base, traffic, stop_mask, traffic_cfg.overlay_alpha)
        empty = np.zeros((0, 2), dtype=np.float32)
        return render.draw_trajectories(
            base, visual["trajectories"], decision["mode_scores"],
            empty, np.zeros(0, dtype=np.float32), self._bev,
            self._cfg.model.driving.bev.fov_deg, draw_gt=False)


def _panel_row(images, labels, tile):
    """把三个 BEV 输出缩放、标注并横向拼为固定宽度。"""
    panels = [
        _title(cv2.resize(image, (tile, tile), interpolation=cv2.INTER_NEAREST), label)
        for image, label in zip(images, labels)
    ]
    return np.ascontiguousarray(np.hstack(panels))


def _camera_panel(frame, width, height, decision, observation, command):
    """相机等比缩放到首行中央，并叠加本次决策与车辆状态 HUD。"""
    canvas = _letterbox(frame, width, height)
    lines = (
        "mode {}  history {}".format(
            decision["mode"], "0.5s" if decision["history_valid"] else "warming"),
        "score {:.2f}  conf {:.2f}  behavior {}:{:.2f}".format(
            decision["mode_scores"][decision["mode"]],
            decision["confidence"][decision["mode"]],
            int(np.argmax(decision["behavior_probabilities"])),
            float(np.max(decision["behavior_probabilities"]))),
        "speed {:.2f}  target {:.2f} m/s".format(
            observation["speed_mps"], command["target_speed_mps"]),
        "thr {:.2f}  steer {:.2f}  brake {:.2f}".format(
            command["throttle"], command["steer"], command["brake"]),
    )
    return _annotate(canvas, lines)


def _letterbox(image, width, height):
    """保持宽高比缩放并居中填充到目标画布。"""
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image, (max(int(round(image.shape[1] * scale)), 1),
                max(int(round(image.shape[0] * scale)), 1)))
    canvas = np.full((height, width, 3), _BACKGROUND, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def _title(image, text):
    """使用 OpenCV 内置字体给推理面板加英文标题。"""
    output = image.copy()
    cv2.putText(output, text, (7, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(output, text, (7, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def _annotate(image, lines):
    """给相机首行叠加不依赖中文字体的决策摘要。"""
    output = image.copy()
    for index, line in enumerate(lines):
        y = 24 + index * 23
        cv2.putText(output, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(output, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return output
