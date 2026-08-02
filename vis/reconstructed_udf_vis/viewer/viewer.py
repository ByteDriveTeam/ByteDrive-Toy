"""Open3D 稀疏 TUDF 查看器：逐帧、全局/自车 BEV、着色和截图。

模块: vis/reconstructed_udf_vis/viewer/viewer.py
依赖: datetime, pathlib, time, numpy, open3d, reconstructed_pointcloud_vis,
      reconstructed_udf_vis.render/viewer.checks
读取配置: reconstructed_udf_vis 全树；model.driving.bev
对外接口:
    - UdfViewer(data, cfg, bev_cfg).run() -> None
"""

from datetime import datetime
from pathlib import Path
import time

import numpy as np
import open3d as o3d

from vis.reconstructed_pointcloud_vis.render import current_bev_center, current_bev_mask
from vis.reconstructed_udf_vis.render import RenderState, render_trajectories, render_voxels
from vis.reconstructed_udf_vis.viewer.checks.viewer_checks import check_viewer_inputs

_REPO_ROOT = Path(__file__).resolve().parents[3]
_COLOR_MODES = ("udf", "weight", "semantic", "source", "actor")


class UdfViewer:
    """管理稀疏 TUDF 的静动态逐帧组合显示。"""

    def __init__(self, data, cfg, bev_cfg):
        screenshot = Path(cfg.screenshot_dir)
        self._screenshot_dir = (screenshot if screenshot.is_absolute()
                                else _REPO_ROOT / screenshot).resolve()
        check_viewer_inputs(data, self._screenshot_dir, _REPO_ROOT)
        self._data, self._cfg, self._bev_cfg = data, cfg, bev_cfg
        self._state = RenderState(
            cfg.initial_show_static, cfg.initial_show_dynamic,
            cfg.initial_show_trajectory, cfg.initial_follow_ego, 0,
            cfg.initial_color_mode)
        self._playing, self._last_tick = False, time.perf_counter()
        self._vis = o3d.visualization.VisualizerWithKeyCallback()
        self._cloud = render_voxels(data, self._state, cfg, bev_cfg)
        self._full_trajectory = render_trajectories(data)
        self._trajectory = self._trajectory_geometry()
        self._ego = self._ego_geometry()

    def run(self):
        created = self._vis.create_window(
            self._cfg.window_name, self._cfg.width, self._cfg.height)
        if not created:
            raise RuntimeError("Open3D TUDF 窗口创建失败")
        for geometry in (self._cloud, self._trajectory, self._ego):
            self._vis.add_geometry(geometry, reset_bounding_box=geometry is self._cloud)
        self._configure_view()
        self._register_keys()
        self._vis.register_animation_callback(self._animation)
        self._print_help()
        self._print_status()
        try:
            self._vis.run()
        finally:
            self._vis.destroy_window()

    def _configure_view(self):
        option = self._vis.get_render_option()
        option.background_color = np.asarray(self._cfg.background_rgb, dtype=np.float64) / 255
        option.point_size = float(self._cfg.point_size)
        option.line_width = float(self._cfg.trajectory_line_width)
        control = self._vis.get_view_control()
        control.set_lookat(self._lookat())
        control.set_front(np.asarray(self._cfg.camera.front, dtype=np.float64))
        control.set_up(np.asarray(self._cfg.camera.up, dtype=np.float64))
        control.set_zoom(float(self._cfg.camera.zoom))

    def _register_keys(self):
        callbacks = {
            ord("Q"): lambda vis: vis.close(), ord(" "): self._toggle_play,
            ord("S"): lambda vis: self._toggle("show_static"),
            ord("D"): lambda vis: self._toggle("show_dynamic"),
            ord("T"): lambda vis: self._toggle("show_trajectory"),
            ord("C"): self._cycle_color, ord("G"): self._toggle_follow,
            ord("R"): self._reset_camera, ord("W"): self._screenshot,
            ord("H"): lambda vis: self._print_help() or False,
            ord("["): lambda vis: self._step(-1), ord("]"): lambda vis: self._step(1),
            263: lambda vis: self._step(-1), 262: lambda vis: self._step(1),
        }
        for key, callback in callbacks.items():
            self._vis.register_key_callback(key, callback)

    def _refresh(self):
        _assign_cloud(self._cloud, render_voxels(
            self._data, self._state, self._cfg, self._bev_cfg))
        _assign_lines(self._trajectory, self._trajectory_geometry())
        _assign_mesh(self._ego, self._ego_geometry())
        for geometry in (self._cloud, self._trajectory, self._ego):
            self._vis.update_geometry(geometry)
        if self._state.follow_ego:
            self._vis.get_view_control().set_lookat(self._lookat())
        self._vis.update_renderer()
        self._print_status()
        return False

    def _trajectory_geometry(self):
        if not self._state.show_trajectory:
            return o3d.geometry.LineSet()
        if not self._state.follow_ego:
            return o3d.geometry.LineSet(self._full_trajectory)
        points, lines = np.asarray(self._full_trajectory.points), np.asarray(
            self._full_trajectory.lines)
        if not len(lines):
            return o3d.geometry.LineSet()
        inside = current_bev_mask(
            points, self._data.ego_pose[self._state.frame_index], self._bev_cfg)
        keep = inside[lines].all(axis=1)
        selected = lines[keep]
        if not len(selected):
            return o3d.geometry.LineSet()
        used, inverse = np.unique(selected, return_inverse=True)
        result = o3d.geometry.LineSet()
        result.points = o3d.utility.Vector3dVector(points[used])
        result.lines = o3d.utility.Vector2iVector(inverse.reshape(-1, 2).astype(np.int32))
        result.colors = o3d.utility.Vector3dVector(
            np.asarray(self._full_trajectory.colors)[keep])
        return result

    def _ego_geometry(self):
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=self._cfg.coordinate_frame_size_m)
        mesh.transform(_pose_matrix(self._data.ego_pose[self._state.frame_index].numpy()))
        return mesh

    def _lookat(self):
        pose = self._data.ego_pose[self._state.frame_index]
        return current_bev_center(pose, self._bev_cfg) \
            if self._state.follow_ego else self._data.center.astype(np.float64)

    def _toggle(self, name):
        setattr(self._state, name, not getattr(self._state, name))
        return self._refresh()

    def _step(self, amount):
        self._state.frame_index = (self._state.frame_index + amount) % self._data.num_frames
        return self._refresh()

    def _animation(self, _vis):
        if not self._playing:
            return False
        now = time.perf_counter()
        if now - self._last_tick < 1 / self._cfg.play_fps:
            return False
        self._last_tick = now
        return self._step(1)

    def _toggle_play(self, _vis):
        self._playing = not self._playing
        self._last_tick = time.perf_counter()
        return False

    def _cycle_color(self, _vis):
        index = (_COLOR_MODES.index(self._state.color_mode) + 1) % len(_COLOR_MODES)
        self._state.color_mode = _COLOR_MODES[index]
        return self._refresh()

    def _toggle_follow(self, _vis):
        self._state.follow_ego = not self._state.follow_ego
        result = self._refresh()
        self._configure_view()
        return result

    def _reset_camera(self, _vis):
        self._configure_view()
        return False

    def _screenshot(self, _vis):
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        path = self._screenshot_dir / "{}_f{:06d}_{}_{}.png".format(
            self._data.scene_name, self._state.frame_index, self._state.color_mode, stamp)
        self._vis.capture_screen_image(str(path), do_render=True)
        print("[稀疏TUDF] 截图已保存:", path)
        return False

    def _print_help(self):
        print("[稀疏TUDF] Space 播放 | [/] 切帧 | S/D/T 图层 | C 着色 | G 全局/自车BEV")
        print("[稀疏TUDF] R 相机 | W 截图 | H 帮助 | Q 退出")

    def _print_status(self):
        scope = "全局" if not self._state.follow_ego else "自车BEV"
        print("[稀疏TUDF] 场景={} 帧={}/{} 着色={} 范围={} 播放={}".format(
            self._data.scene_name, self._state.frame_index, self._data.num_frames - 1,
            self._state.color_mode, scope, "开" if self._playing else "关"))


def _assign_cloud(target, source):
    target.points, target.colors = source.points, source.colors


def _assign_lines(target, source):
    target.points, target.lines, target.colors = source.points, source.lines, source.colors


def _assign_mesh(target, source):
    target.vertices, target.triangles = source.vertices, source.triangles
    target.vertex_colors, target.vertex_normals = source.vertex_colors, source.vertex_normals


def _pose_matrix(pose):
    x, y, z, roll, pitch, yaw = pose
    roll, pitch, yaw = np.deg2rad((roll, pitch, yaw))
    cr, cp, cy = np.cos((roll, pitch, yaw))
    sr, sp, sy = np.sin((roll, pitch, yaw))
    matrix = np.eye(4)
    matrix[:3, 3] = (x, y, z)
    matrix[:3, :3] = (
        (cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr),
        (sp, -cp * sr, cp * cr))
    return matrix


__all__ = ["UdfViewer"]
