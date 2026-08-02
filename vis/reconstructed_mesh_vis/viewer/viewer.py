"""Open3D Mesh 查看器：逐帧播放、全局/自车 BEV、着色与截图。

模块: vis/reconstructed_mesh_vis/viewer/viewer.py
依赖: datetime, pathlib, time, numpy, open3d,
      vis.reconstructed_mesh_vis.render/viewer.checks
读取配置: reconstructed_mesh_vis 全树；model.driving.bev.x/y_min/max_m
对外接口:
    - MeshViewer(data, cfg, bev_cfg).run() -> None
"""

from datetime import datetime
from pathlib import Path
import time

import numpy as np
import open3d as o3d

from vis.reconstructed_mesh_vis.render import (
    RenderState,
    render_dynamic_mesh,
    render_static_mesh,
    render_trajectories,
)
from vis.reconstructed_mesh_vis.viewer.checks.viewer_checks import check_viewer_inputs
from vis.reconstructed_pointcloud_vis.render import current_bev_center, current_bev_mask

_REPO_ROOT = Path(__file__).resolve().parents[3]
_COLOR_MODES = ("semantic", "source", "actor", "method")


class MeshViewer:
    """管理一个重建场景的逐帧 Open3D 交互窗口。"""

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
        self._playing = False
        self._wireframe = False
        self._last_tick = time.perf_counter()
        self._vis = o3d.visualization.VisualizerWithKeyCallback()
        self._static = render_static_mesh(data, self._state, cfg, bev_cfg)
        self._dynamic = render_dynamic_mesh(data, self._state, cfg, bev_cfg)
        self._full_trajectory = render_trajectories(data)
        self._trajectory = self._trajectory_geometry()
        self._ego = self._ego_geometry()

    def run(self):
        """打开阻塞式 Open3D 查看器，直至用户退出。"""
        created = self._vis.create_window(
            self._cfg.window_name, self._cfg.width, self._cfg.height)
        if not created:
            raise RuntimeError("Open3D Mesh 窗口创建失败；请确认图形桌面可用")
        for geometry in (self._static, self._dynamic, self._trajectory, self._ego):
            self._vis.add_geometry(geometry, reset_bounding_box=geometry is self._static)
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
        option.line_width = float(self._cfg.trajectory_line_width)
        option.mesh_show_wireframe = self._wireframe
        control = self._vis.get_view_control()
        control.set_lookat(self._lookat())
        control.set_front(np.asarray(self._cfg.camera.front, dtype=np.float64))
        control.set_up(np.asarray(self._cfg.camera.up, dtype=np.float64))
        control.set_zoom(float(self._cfg.camera.zoom))

    def _register_keys(self):
        callbacks = {
            ord("Q"): self._quit,
            ord(" "): self._toggle_play,
            ord("S"): lambda vis: self._toggle("show_static"),
            ord("D"): lambda vis: self._toggle("show_dynamic"),
            ord("T"): lambda vis: self._toggle("show_trajectory"),
            ord("C"): self._cycle_color,
            ord("F"): self._toggle_wireframe,
            ord("G"): self._toggle_follow,
            ord("R"): self._reset_camera,
            ord("W"): self._screenshot,
            ord("H"): self._help_callback,
            ord("["): lambda vis: self._step(-1),
            ord("]"): lambda vis: self._step(1),
            263: lambda vis: self._step(-1),
            262: lambda vis: self._step(1),
        }
        for key, callback in callbacks.items():
            self._vis.register_key_callback(key, callback)

    def _refresh(self):
        _assign_mesh(self._static, render_static_mesh(
            self._data, self._state, self._cfg, self._bev_cfg))
        _assign_mesh(self._dynamic, render_dynamic_mesh(
            self._data, self._state, self._cfg, self._bev_cfg))
        _assign_lines(self._trajectory, self._trajectory_geometry())
        _assign_mesh(self._ego, self._ego_geometry())
        for geometry in (self._static, self._dynamic, self._trajectory, self._ego):
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
        points = np.asarray(self._full_trajectory.points)
        lines = np.asarray(self._full_trajectory.lines)
        if not len(lines):
            return o3d.geometry.LineSet()
        inside = current_bev_mask(
            points, self._data.ego_pose[self._state.frame_index], self._bev_cfg)
        keep_lines = inside[lines].all(axis=1)
        kept = lines[keep_lines]
        if not len(kept):
            return o3d.geometry.LineSet()
        used, inverse = np.unique(kept, return_inverse=True)
        cropped = o3d.geometry.LineSet()
        cropped.points = o3d.utility.Vector3dVector(points[used])
        cropped.lines = o3d.utility.Vector2iVector(inverse.reshape(-1, 2).astype(np.int32))
        cropped.colors = o3d.utility.Vector3dVector(
            np.asarray(self._full_trajectory.colors)[keep_lines])
        return cropped

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

    def _step(self, step):
        self._state.frame_index = (self._state.frame_index + step) % self._data.num_frames
        return self._refresh()

    def _animation(self, _vis):
        if not self._playing:
            return False
        now = time.perf_counter()
        if now - self._last_tick < 1.0 / self._cfg.play_fps:
            return False
        self._last_tick = now
        return self._step(1)

    def _toggle_play(self, _vis):
        self._playing = not self._playing
        self._last_tick = time.perf_counter()
        self._print_status()
        return False

    def _cycle_color(self, _vis):
        index = (_COLOR_MODES.index(self._state.color_mode) + 1) % len(_COLOR_MODES)
        self._state.color_mode = _COLOR_MODES[index]
        return self._refresh()

    def _toggle_wireframe(self, _vis):
        self._wireframe = not self._wireframe
        self._vis.get_render_option().mesh_show_wireframe = self._wireframe
        self._vis.update_renderer()
        self._print_status()
        return False

    def _toggle_follow(self, _vis):
        self._state.follow_ego = not self._state.follow_ego
        result = self._refresh()
        self._configure_view()
        self._vis.update_renderer()
        return result

    def _reset_camera(self, _vis):
        self._configure_view()
        self._vis.update_renderer()
        return False

    def _screenshot(self, _vis):
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        safe_scene = "".join(character if character.isalnum() or character in "-_." else "_"
                             for character in self._data.scene_name)
        path = self._screenshot_dir / "{}_f{:06d}_{}_{}.png".format(
            safe_scene, self._state.frame_index, self._state.color_mode, stamp)
        self._vis.capture_screen_image(str(path), do_render=True)
        print("[重建Mesh] 截图已保存:", path)
        return False

    def _quit(self, _vis):
        self._vis.close()
        return False

    def _help_callback(self, _vis):
        self._print_help()
        return False

    def _print_help(self):
        print("[重建Mesh] Space 播放/暂停 | [/] 或 ←/→ 切帧 | S 静态 | D 动态 | T 轨迹")
        print("[重建Mesh] C 着色 | F 线框 | G 全局/自车BEV | R 相机 | W 截图 | H 帮助 | Q 退出")

    def _print_status(self):
        scope = "全局" if not self._state.follow_ego \
            else "自车BEV x[{:.1f},{:.1f}] y[{:.1f},{:.1f}]m".format(
                self._bev_cfg.x_min_m, self._bev_cfg.x_max_m,
                self._bev_cfg.y_min_m, self._bev_cfg.y_max_m)
        print("[重建Mesh] 场景={} | 帧={}/{} | actor={} Mesh={} | 着色={} | 播放={} 范围={} 线框={}".format(
            self._data.scene_name, self._state.frame_index, self._data.num_frames - 1,
            self._data.num_objects, self._data.num_meshed_objects, self._state.color_mode,
            "开" if self._playing else "关", scope,
            "开" if self._wireframe else "关"))


def _assign_mesh(target, source):
    target.vertices, target.triangles = source.vertices, source.triangles
    target.vertex_normals, target.vertex_colors = source.vertex_normals, source.vertex_colors


def _assign_lines(target, source):
    target.points, target.lines, target.colors = source.points, source.lines, source.colors


def _pose_matrix(pose):
    x, y, z, roll, pitch, yaw = pose
    roll, pitch, yaw = np.deg2rad((roll, pitch, yaw))
    cr, cp, cy = np.cos((roll, pitch, yaw))
    sr, sp, sy = np.sin((roll, pitch, yaw))
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = (x, y, z)
    matrix[:3, :3] = (
        (cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr),
        (sp, -cp * sr, cp * cr))
    return matrix


__all__ = ["MeshViewer"]
