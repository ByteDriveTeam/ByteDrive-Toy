"""Open3D 交互查看器：切换全局/当前帧 BEV、静动态层、轨迹与着色并保存截图。

模块: vis/reconstructed_pointcloud_vis/viewer/viewer.py
依赖: datetime, pathlib, numpy, open3d, vis.reconstructed_pointcloud_vis.render、
      vis.reconstructed_pointcloud_vis.viewer.checks
读取配置: reconstructed_pointcloud_vis.window_name/width/height/point_size/background_rgb、
          coordinate_frame_size_m/trajectory_line_width/screenshot_dir/initial_*/camera.*；
          model.driving.bev.x_min_m/x_max_m/y_min_m/y_max_m；
          点数与颜色配置经 render_pointcloud 透传
对外接口:
    - PointcloudViewer(data, cfg, bev_cfg).run() -> None
说明: 使用 Open3D legacy VisualizerWithKeyCallback，以兼容 0.19 的原生相机交互与大点云渲染。
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import open3d as o3d

from vis.reconstructed_pointcloud_vis.render import (
    RenderState,
    current_bev_center,
    current_bev_mask,
    render_pointcloud,
    render_trajectories,
)
from vis.reconstructed_pointcloud_vis.viewer.checks import check_viewer_inputs

_REPO_ROOT = Path(__file__).resolve().parents[3]
_COLOR_MODES = ("semantic", "source", "actor", "height")


class PointcloudViewer:
    """管理一个融合场景的 Open3D 交互窗口。"""

    def __init__(self, data, cfg, bev_cfg):
        screenshot_dir = Path(cfg.screenshot_dir)
        self._screenshot_dir = (screenshot_dir if screenshot_dir.is_absolute()
                                else _REPO_ROOT / screenshot_dir).resolve()
        check_viewer_inputs(data, self._screenshot_dir, _REPO_ROOT)
        self._data = data
        self._cfg = cfg
        self._bev_cfg = bev_cfg
        self._frame_position = 0
        self._state = RenderState(
            show_static=cfg.initial_show_static,
            show_dynamic=cfg.initial_show_dynamic,
            show_trajectory=cfg.initial_show_trajectory,
            all_dynamic_frames=cfg.initial_all_dynamic_frames,
            frame_index=0,
            color_mode=cfg.initial_color_mode,
            spatial_scope=cfg.initial_spatial_scope,
        )
        self._vis = o3d.visualization.VisualizerWithKeyCallback()
        self._cloud = render_pointcloud(data, self._state, cfg, bev_cfg)
        self._full_trajectory = render_trajectories(data)
        self._trajectory = self._trajectory_geometry()
        self._ego_frame = self._ego_frame_geometry()

    def run(self):
        """打开阻塞式 Open3D 窗口，直至按 Q/Esc 或关闭窗口。"""
        created = self._vis.create_window(
            window_name=self._cfg.window_name,
            width=self._cfg.width,
            height=self._cfg.height,
        )
        if not created:
            raise RuntimeError("Open3D 窗口创建失败；请确认当前会话有可用图形桌面")
        self._vis.add_geometry(self._cloud, reset_bounding_box=True)
        self._vis.add_geometry(self._trajectory, reset_bounding_box=False)
        self._vis.add_geometry(self._ego_frame, reset_bounding_box=False)
        self._configure_view()
        self._register_keys()
        self._print_help()
        self._print_status()
        try:
            self._vis.run()
        finally:
            self._vis.destroy_window()

    def _configure_view(self):
        option = self._vis.get_render_option()
        option.background_color = np.asarray(self._cfg.background_rgb, dtype=np.float64) / 255.0
        option.point_size = float(self._cfg.point_size)
        option.line_width = float(self._cfg.trajectory_line_width)
        control = self._vis.get_view_control()
        control.set_lookat(self._view_center())
        control.set_front(np.asarray(self._cfg.camera.front, dtype=np.float64))
        control.set_up(np.asarray(self._cfg.camera.up, dtype=np.float64))
        control.set_zoom(float(self._cfg.camera.zoom))

    def _register_keys(self):
        callbacks = {
            ord("Q"): self._quit,
            ord("S"): lambda vis: self._toggle("show_static"),
            ord("D"): lambda vis: self._toggle("show_dynamic"),
            ord("T"): lambda vis: self._toggle("show_trajectory"),
            ord("A"): lambda vis: self._toggle("all_dynamic_frames"),
            ord("B"): self._toggle_spatial_scope,
            ord("C"): self._cycle_color,
            ord("R"): self._reset_camera,
            ord("W"): self._screenshot,
            ord("H"): self._help_callback,
            ord("["): lambda vis: self._step_frame(-1),
            ord("]"): lambda vis: self._step_frame(1),
            263: lambda vis: self._step_frame(-1),
            262: lambda vis: self._step_frame(1),
        }
        for key, callback in callbacks.items():
            self._vis.register_key_callback(key, callback)

    def _refresh(self):
        updated = render_pointcloud(self._data, self._state, self._cfg, self._bev_cfg)
        self._cloud.points = updated.points
        self._cloud.colors = updated.colors
        trajectory = self._trajectory_geometry()
        self._trajectory.points = trajectory.points
        self._trajectory.lines = trajectory.lines
        self._trajectory.colors = trajectory.colors
        ego_frame = self._ego_frame_geometry()
        self._ego_frame.vertices = ego_frame.vertices
        self._ego_frame.triangles = ego_frame.triangles
        self._ego_frame.vertex_colors = ego_frame.vertex_colors
        self._ego_frame.vertex_normals = ego_frame.vertex_normals
        self._ego_frame.triangle_normals = ego_frame.triangle_normals
        self._vis.update_geometry(self._cloud)
        self._vis.update_geometry(self._trajectory)
        self._vis.update_geometry(self._ego_frame)
        if self._state.spatial_scope == "bev":
            self._vis.get_view_control().set_lookat(self._view_center())
        self._vis.update_renderer()
        self._print_status()
        return False

    def _trajectory_geometry(self):
        if not self._state.show_trajectory:
            return o3d.geometry.LineSet()
        if self._state.spatial_scope == "global":
            return o3d.geometry.LineSet(self._full_trajectory)
        points = np.asarray(self._full_trajectory.points)
        lines = np.asarray(self._full_trajectory.lines)
        if not len(lines):
            return o3d.geometry.LineSet()
        inside = current_bev_mask(
            points, self._data.ego_pose[self._state.frame_index], self._bev_cfg)
        kept = lines[inside[lines].all(axis=1)]
        if not len(kept):
            return o3d.geometry.LineSet()
        used, inverse = np.unique(kept, return_inverse=True)
        cropped = o3d.geometry.LineSet()
        cropped.points = o3d.utility.Vector3dVector(points[used])
        cropped.lines = o3d.utility.Vector2iVector(inverse.reshape(-1, 2).astype(np.int32))
        colors = np.asarray(self._full_trajectory.colors)
        keep_lines = inside[lines].all(axis=1)
        cropped.colors = o3d.utility.Vector3dVector(colors[keep_lines])
        return cropped

    def _ego_frame_geometry(self):
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=self._cfg.coordinate_frame_size_m)
        frame.transform(_pose_matrix(self._data.ego_pose[self._state.frame_index]))
        return frame

    def _view_center(self):
        pose = self._data.ego_pose[self._state.frame_index]
        return current_bev_center(pose, self._bev_cfg) \
            if self._state.spatial_scope == "bev" else self._data.center.astype(np.float64)

    def _toggle(self, name):
        setattr(self._state, name, not getattr(self._state, name))
        return self._refresh()

    def _step_frame(self, step):
        self._frame_position = (self._frame_position + step) % self._data.num_frames
        self._state.frame_index = int(self._data.frame_indices[self._frame_position])
        return self._refresh()

    def _toggle_spatial_scope(self, _vis):
        self._state.spatial_scope = "bev" \
            if self._state.spatial_scope == "global" else "global"
        result = self._refresh()
        self._configure_view()
        self._vis.update_renderer()
        return result

    def _cycle_color(self, _vis):
        position = (_COLOR_MODES.index(self._state.color_mode) + 1) % len(_COLOR_MODES)
        self._state.color_mode = _COLOR_MODES[position]
        return self._refresh()

    def _reset_camera(self, _vis):
        self._configure_view()
        self._vis.update_renderer()
        return False

    def _screenshot(self, _vis):
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)
        frame = "all" if self._state.all_dynamic_frames else "f{:06d}".format(
            self._state.frame_index)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        safe_scene = "".join(character if character.isalnum() or character in "-_." else "_"
                             for character in self._data.scene_name)
        path = self._screenshot_dir / "{}_{}_{}_{}_{}.png".format(
            safe_scene, self._state.spatial_scope, frame, self._state.color_mode, stamp)
        self._vis.capture_screen_image(str(path), do_render=True)
        print("[重建点云] 截图已保存:", path)
        return False

    def _quit(self, _vis):
        self._vis.close()
        return False

    def _help_callback(self, _vis):
        self._print_help()
        return False

    def _print_help(self):
        print("[重建点云] 热键: [ / ] 或 ← / → 切帧 | B 全局/当前帧 BEV | A 动态全帧/当前帧")
        print("[重建点云]       S 静态 | D 动态 | T 轨迹 | C 着色 | R 重置相机 | W 截图 | H 帮助 | Q 退出")
        print("[重建点云] 鼠标: 左键旋转 | Ctrl+左键平移 | 滚轮缩放")

    def _print_status(self):
        dynamic_time = "全部帧" if self._state.all_dynamic_frames else "当前帧"
        scope = "全局" if self._state.spatial_scope == "global" \
            else "BEV x[{:.1f},{:.1f}] y[{:.1f},{:.1f}]m".format(
                self._bev_cfg.x_min_m, self._bev_cfg.x_max_m,
                self._bev_cfg.y_min_m, self._bev_cfg.y_max_m)
        print("[重建点云] 场景={} | 点={:,} (静态 {:,}, 动态 {:,}) | 帧={}/{} | 范围={} | 动态={} | 着色={} | S={} D={} T={}".format(
            self._data.scene_name, self._data.num_points, self._data.num_static,
            self._data.num_dynamic, self._state.frame_index, self._data.num_frames - 1,
            scope, dynamic_time, self._state.color_mode,
            "开" if self._state.show_static else "关",
            "开" if self._state.show_dynamic else "关",
            "开" if self._state.show_trajectory else "关"))


def _pose_matrix(pose):
    x, y, z, roll, pitch, yaw = np.asarray(pose, dtype=np.float64)
    roll, pitch, yaw = np.deg2rad((roll, pitch, yaw))
    cr, cp, cy = np.cos((roll, pitch, yaw))
    sr, sp, sy = np.sin((roll, pitch, yaw))
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = (x, y, z)
    matrix[:3, :3] = (
        (cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr),
        (sp, -cp * sr, cp * cr),
    )
    return matrix


__all__ = ["PointcloudViewer"]
