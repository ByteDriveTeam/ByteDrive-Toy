"""渲染：3D 框投影到 RGB、深度/语义/光流着色、lidar+框 鸟瞰图、多面板合成与 HUD。

模块: vis/data_vis/draw/draw.py
依赖: cv2, numpy, vis.data_vis.geometry, vis.data_vis.palette
读取配置: cfg.data_vis.bbox(thickness/max_distance_m/draw_static/colors)、cfg.data_vis.depth(max_display_m/colormap)、
          cfg.data_vis.optical_flow(max_flow)、cfg.data_vis.bev(range_m/size_px/point_radius/color_by/bg)、
          cfg.data_vis.lidar(max_points_draw)、cfg.data_vis.traffic_lights(nearest_count/selected_color/
          selected_line_thickness/selected_point_radius)
对外接口:
    - render_frame(frame, meta, vcfg, state) -> np.ndarray   # 合成单帧的完整 BGR 画布（未缩放）
说明: 投影约定见 vis/data_vis/geometry。每行一种相机模态、每列一个相机；无视觉层时改用状态 BEV 绘制自车、
      车辆/行人轨迹与 Box。仅渲染 state["available"] 为真且对应开关开启的层，故任意采集开关组合都自适应。
      面板间留间隔、HUD 半透明叠加并只列可用模态的开关，提升可读性。点云用向量化散点（规范 §9），逐物体
      绘框因含 cv2 副作用而用循环。
"""

import cv2
import numpy as np

from vis.data_vis import geometry as g
from vis.data_vis.palette import flow_to_bgr, tag_to_bgr

# colormap 名 -> OpenCV 常量（须与 config/schema._DATA_VIS_COLORMAPS 保持一致）
_COLORMAPS = {"turbo": cv2.COLORMAP_TURBO, "jet": cv2.COLORMAP_JET, "magma": cv2.COLORMAP_MAGMA,
              "viridis": cv2.COLORMAP_VIRIDIS, "plasma": cv2.COLORMAP_PLASMA,
              "inferno": cv2.COLORMAP_INFERNO}

# 12 条棱的端点索引，对应 geometry._CORNER_SIGNS 的 8 角点顺序
_EDGES = np.array([(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                   (0, 4), (1, 5), (2, 6), (3, 7)])
# BEV 顶面 4 角（+z 面）构成俯视矩形轮廓
_TOP_FACE = [4, 5, 6, 7]
# 相机显示顺序（从左到右）；未列出的名字排在最后
_CAM_ORDER = ["front_left", "left", "front", "front_right", "right", "back"]
_NEAR_M = 0.2          # 近平面：前向距离小于此值的角点不参与连线（避免投影发散）
_DEFAULT_BGR = (200, 200, 200)
_GUTTER = 3            # 面板/行之间的间隔像素，分隔视觉、避免相邻画面糊在一起
_BG = (28, 28, 30)     # 间隔条与占位面板的暗背景


def render_frame(frame, meta, vcfg, state):
    """把一帧数据合成为单张 BGR 画布（仅渲染存在且开启的传感器层）。

    参数:
        frame: SceneReader.frame(i) 的返回（rgb/depth/semantic/optical_flow/lidar/ego/bboxes）
        meta:  场景级 meta（intrinsics/extrinsics/static_bboxes/camera_names 等）
        vcfg:  cfg.data_vis 配置对象
        state: dict 显示开关与上下文，含 available（各模态是否落盘）、show_*、idx、num_frames、playing
    返回:
        合成后的 BGR 画布（未按 display.scale 缩放）
    """
    cams = order_cameras(meta["camera_names"])
    avail = state["available"]
    ego_pose = frame["ego"]["transform"]
    boxes = frame["bboxes"] + (meta["static_bboxes"] if state["show_static"] else [])

    rows = []
    if avail["rgb"] and state["show_rgb"]:
        rows.append(_rgb_row(frame["rgb"], boxes, ego_pose, meta, cams, vcfg.bbox, state["show_bbox"]))
    if avail["depth"] and state["show_depth"]:
        rows.append(_modality_row(cams, "depth ", lambda c: _colorize_depth(frame["depth"][c], vcfg.depth)))
    if avail["semantic"] and state["show_semantic"]:
        rows.append(_modality_row(cams, "seg ", lambda c: tag_to_bgr(frame["semantic"][c])))
    if avail["optical_flow"] and state["show_flow"]:
        rows.append(_modality_row(cams, "flow ", lambda c: _colorize_flow(frame["optical_flow"][c],
                                                                          vcfg.optical_flow)))
    left = _stack_rows(rows) if rows else _state_bev(
        boxes, ego_pose, state.get("trajectory", []), state.get("actor_trajectories", {}),
        frame["meta"].get("relevant_traffic_control"), vcfg)

    canvas = left
    if avail["lidar"] and state["show_bev"]:
        bev = _render_bev(
            frame["lidar"], boxes, ego_pose, meta["lidar_extrinsic"],
            frame["meta"].get("relevant_traffic_control"), vcfg)
        canvas = _join_lr(left, _fit_height(bev, left.shape[0]))

    traffic_light_lines = _traffic_light_hud_lines(
        frame.get("traffic_light_states", []), meta.get("traffic_lights", []), ego_pose,
        frame["meta"].get("relevant_traffic_control"), vcfg.traffic_lights.nearest_count)
    _put_hud(canvas, frame["meta"], state, traffic_light_lines, avail)
    return canvas


def order_cameras(names):
    """按从左到右的显示习惯排序相机名（未知名排末尾，保持稳定）。"""
    return sorted(names, key=lambda n: _CAM_ORDER.index(n) if n in _CAM_ORDER else len(_CAM_ORDER))


def _camera_intrinsics(intrinsics, camera):
    """读取指定相机内参；兼容旧数据集中所有相机共享单个内参 dict 的格式。"""
    return intrinsics if isinstance(intrinsics.get("fx"), (int, float)) else intrinsics[camera]


def _panel_hw(meta, cams):
    """相机面板的 (高,宽)：优先用内参记录的分辨率，缺则由主点推回（cx,cy 即半幅）。"""
    intr = _camera_intrinsics(meta["intrinsics"], cams[0])
    return int(intr.get("height", intr["cy"] * 2)), int(intr.get("width", intr["cx"] * 2))


# ---------- 相机行（每行一种模态、每列一个相机）----------

def _rgb_row(rgb, boxes, ego_pose, meta, cams, bcfg, show_bbox):
    """RGB 行：每相机用自身内参在画面上叠投影框；某相机无帧则以黑底占位。"""
    hw = _panel_hw(meta, cams)
    panels = [_rgb_panel(rgb.get(c), boxes, ego_pose, meta["extrinsics"][c],
                         g.intrinsic_matrix(_camera_intrinsics(meta["intrinsics"], c)),
                         bcfg, show_bbox, c, hw) for c in cams]
    return _hrow(panels)


def _modality_row(cams, prefix, colorize):
    """非 RGB 模态行：对每相机调用 colorize(cam) 得到 BGR 面板并加标题后横排。"""
    return _hrow([_titled(colorize(c), prefix + c) for c in cams])


def _placeholder(meta, cams):
    """所有相机层均关闭时的占位面板（保证画布非空，BEV 仍可拼接其右）。"""
    h, w = _panel_hw(meta, cams)
    return _titled(np.full((h, w, 3), _BG, np.uint8), "(camera layers off)")


def _state_bev(boxes, ego_pose, trajectory, actor_trajectories, relevant_control, vcfg):
    """无视觉传感器时用 Agent 轨迹与 Box 生成可读的状态鸟瞰图。"""
    bev = vcfg.bev
    size, rng = bev.size_px, bev.range_m
    canvas = np.full((size, size, 3), bev.bg, dtype=np.uint8)
    scale, center = size / (2.0 * rng), size * 0.5
    w2e = g.world_to_ego(ego_pose)
    if trajectory:
        points = g.transform_points(np.asarray([[pose[0], pose[1], 0.0] for pose in trajectory]), w2e)
        _draw_track(canvas, points, center, scale, rng, (80, 220, 255), 2)
    box_colors = {int(box["id"]): tuple(vcfg.bbox.colors.get(box["semantic"], _DEFAULT_BGR))
                  for box in boxes if box.get("id") is not None}
    for actor_id, locations in actor_trajectories.items():
        points = g.transform_points(np.asarray(locations, dtype=np.float64), w2e)
        _draw_track(canvas, points, center, scale, rng,
                    box_colors.get(int(actor_id), (150, 150, 150)), 1)
    for box in boxes:
        _draw_box_bev(canvas, box, w2e, center, scale, rng, vcfg.bbox)
    _draw_relevant_control(canvas, relevant_control, w2e, center, scale, rng, vcfg.traffic_lights)
    _draw_ego_marker(canvas, center)
    return _titled(canvas, "agent trajectory / state BEV")


def _draw_track(canvas, points, center, scale, rng, color, thickness):
    """先连接完整轨迹，再由画布裁剪显示范围，避免进出视窗时断线。"""
    if len(points) < 2:
        return
    pixels = np.stack((center + points[:, 1] * scale,
                       center - points[:, 0] * scale), axis=1).astype(np.int32)
    jumps = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1) > max(20.0, rng * 0.5)
    breaks = np.flatnonzero(jumps)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(points) - 1]
    for start, end in zip(starts, ends):
        if end - start >= 1:
            cv2.polylines(canvas, [pixels[start:end + 1]], False, color, thickness, cv2.LINE_AA)


# ---------- RGB 面板与 3D 框投影 ----------

def _rgb_panel(img, boxes, ego_pose, cam_extrinsic, K, bcfg, show_bbox, title, hw):
    panel = img.copy() if img is not None else np.full((hw[0], hw[1], 3), _BG, np.uint8)
    if show_bbox:
        w2c = g.world_to_camera(ego_pose, cam_extrinsic)
        cam_pos = (g.transform_matrix(ego_pose) @ g.transform_matrix(cam_extrinsic))[:3, 3]
        for box in boxes:
            _draw_box_2d(panel, box, w2c, K, cam_pos, bcfg)
    return _titled(panel, title)


def _draw_box_2d(img, box, w2c, K, cam_pos, bcfg):
    """把单个世界系包围框投影并画出可见棱（两端均在相机前方时才连线）。"""
    # 相机就装在主车包围框内，把 ego 框投到自身画面只会占满前景；BEV 仍保留 ego（见 _render_bev）
    if box["semantic"] == "ego":
        return
    if np.linalg.norm(np.subtract(box["location"], cam_pos)) > bcfg.max_distance_m:
        return
    corners = g.bbox_corners(box)
    uv, depth = g.project_points(corners, w2c, K)
    if (depth > _NEAR_M).sum() == 0:  # 整框在相机后方
        return
    uv = uv.astype(np.int32)
    color = tuple(bcfg.colors.get(box["semantic"], _DEFAULT_BGR))
    visible = depth > _NEAR_M
    for a, b in _EDGES:
        if visible[a] and visible[b]:
            cv2.line(img, tuple(uv[a]), tuple(uv[b]), color, bcfg.thickness, cv2.LINE_AA)


# ---------- 深度着色 ----------

def _colorize_depth(depth_m, dcfg):
    """深度（米）-> 伪彩：按 max_display_m 归一截断后套 colormap。"""
    norm = np.clip(depth_m / dcfg.max_display_m, 0.0, 1.0)
    gray = (norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(gray, _COLORMAPS[dcfg.colormap])


# ---------- 光流着色 ----------

def _colorize_flow(flow, fcfg):
    """光流 (H,W,2) 运动矢量 -> BGR：委托 palette.flow_to_bgr（与 pred_vis 共用唯一配色实现）。"""
    return flow_to_bgr(flow[..., 0], flow[..., 1], fcfg.max_flow)


# ---------- 鸟瞰图（主车系俯视）----------

def _render_bev(lidar, boxes, ego_pose, lidar_extrinsic, relevant_control, vcfg):
    """渲染以主车为中心的 BEV：lidar、框、当前相关停止线与主车朝向。"""
    bev = vcfg.bev
    size, rng = bev.size_px, bev.range_m
    canvas = np.full((size, size, 3), bev.bg, dtype=np.uint8)
    scale = size / (2.0 * rng)
    center = size * 0.5

    pts = _lidar_xyz_ego(lidar, lidar_extrinsic)
    colors = _lidar_colors(lidar, pts, bev, vcfg.depth.colormap)
    pts, colors = _subsample(pts, colors, vcfg.lidar.max_points_draw)
    _scatter_bev(canvas, pts, colors, center, scale, rng, bev.point_radius)

    w2e = g.world_to_ego(ego_pose)
    for box in boxes:
        _draw_box_bev(canvas, box, w2e, center, scale, rng, vcfg.bbox)
    _draw_relevant_control(
        canvas, relevant_control, w2e, center, scale, rng, vcfg.traffic_lights)
    _draw_ego_marker(canvas, center)
    return canvas


def _lidar_xyz_ego(lidar, lidar_extrinsic):
    """语义 lidar 结构化点 -> (N,3) 主车系坐标（雷达无旋转，仅平移外参）。"""
    xyz = np.stack([lidar["x"], lidar["y"], lidar["z"]], axis=1).astype(np.float64)
    return xyz + np.asarray(lidar_extrinsic)


def _lidar_colors(lidar, pts_ego, bev, colormap):
    """按配置给每个点上色：tag=语义调色板；height=按主车系高度套 colormap。"""
    if bev.color_by == "tag":
        return tag_to_bgr(lidar["obj_tag"])
    z = pts_ego[:, 2]
    norm = np.clip((z - z.min()) / (np.ptp(z) + 1e-6), 0.0, 1.0)
    gray = (norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(gray, _COLORMAPS[colormap]).reshape(-1, 3)


def _scatter_bev(canvas, pts_ego, colors, center, scale, rng, radius):
    """向量化散点：主车系 (x前,y右) -> 像素 (u右,v上为前)，半径用常量小邻域偏移铺开。"""
    inside = (np.abs(pts_ego[:, 0]) < rng) & (np.abs(pts_ego[:, 1]) < rng)
    x, y, col = pts_ego[inside, 0], pts_ego[inside, 1], colors[inside]
    u = (center + y * scale).astype(np.int32)
    v = (center - x * scale).astype(np.int32)
    size = canvas.shape[0]
    # 邻域偏移是常量级小集合（(2r+1)^2），对每个偏移做整幅向量化写入（规范 §9）
    for du in range(-radius + 1, radius):
        for dv in range(-radius + 1, radius):
            uu, vv = np.clip(u + du, 0, size - 1), np.clip(v + dv, 0, size - 1)
            canvas[vv, uu] = col


def _draw_box_bev(canvas, box, w2e, center, scale, rng, bcfg):
    """把框顶面 4 角投到主车系俯视平面，画矩形轮廓。"""
    corners_ego = g.transform_points(g.bbox_corners(box), w2e)[_TOP_FACE]
    if np.min(np.abs(corners_ego[:, :2])) > rng:  # 完全在视窗外
        return
    u = center + corners_ego[:, 1] * scale
    v = center - corners_ego[:, 0] * scale
    poly = np.stack([u, v], axis=1).astype(np.int32)
    color = tuple(bcfg.colors.get(box["semantic"], _DEFAULT_BGR))
    cv2.polylines(canvas, [poly], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)


def _draw_relevant_control(canvas, control, w2e, center, scale, rng, traffic_cfg):
    """新数据存在有效关联时，在 BEV 画受控车道停止线及其中心点；旧数据保持原样。"""
    if not control or not control.get("valid"):
        return
    location = np.asarray(control["stop_location"], dtype=np.float64)
    yaw = np.radians(float(control["stop_yaw"]))
    half_width = float(control["lane_width"]) * 0.5
    lateral = np.array([-np.sin(yaw), np.cos(yaw), 0.0], dtype=np.float64)
    world_points = np.stack((location - lateral * half_width, location,
                             location + lateral * half_width))
    ego_points = g.transform_points(world_points, w2e)
    if not bool(np.any((np.abs(ego_points[:, 0]) < rng) & (np.abs(ego_points[:, 1]) < rng))):
        return
    pixels = np.stack((center + ego_points[:, 1] * scale,
                       center - ego_points[:, 0] * scale), axis=1).astype(np.int32)
    color = tuple(int(value) for value in traffic_cfg.selected_color)
    cv2.line(canvas, tuple(pixels[0]), tuple(pixels[2]), color,
             traffic_cfg.selected_line_thickness, cv2.LINE_AA)
    cv2.circle(canvas, tuple(pixels[1]), traffic_cfg.selected_point_radius,
               color, thickness=-1, lineType=cv2.LINE_AA)


def _draw_ego_marker(canvas, center):
    """主车标记：朝上的小三角，提示 BEV 的前向（屏幕上方）。"""
    c = int(center)
    tri = np.array([(c, c - 8), (c - 5, c + 6), (c + 5, c + 6)], dtype=np.int32)
    cv2.fillConvexPoly(canvas, tri, (255, 255, 255), cv2.LINE_AA)


def _subsample(pts, colors, max_points):
    """点数超限时等距下采样，保护 BEV 渲染帧率。"""
    if len(pts) <= max_points:
        return pts, colors
    idx = np.linspace(0, len(pts) - 1, max_points).astype(np.intp)
    return pts[idx], colors[idx]


# ---------- HUD 与画布拼接 ----------

_TRAFFIC_LIGHT_STATES = ("red", "yellow", "green", "off", "unknown")


def _traffic_light_hud_lines(traffic_light_states, traffic_lights, ego_pose,
                             relevant_control, nearest_count):
    """生成交通灯 HUD 文本：相关灯、全场统计，以及按触发区距离排序的最近 N 盏灯。"""
    relevant_line = _relevant_control_hud_line(relevant_control)
    if not traffic_light_states:
        return [relevant_line]
    normalized = [item["state"] if item.get("state") in _TRAFFIC_LIGHT_STATES else "unknown"
                  for item in traffic_light_states]
    counts = {name: normalized.count(name) for name in _TRAFFIC_LIGHT_STATES}
    lines = [relevant_line, "TL R:{} Y:{} G:{} O:{} U:{}".format(
        counts["red"], counts["yellow"], counts["green"], counts["off"], counts["unknown"])]
    if nearest_count == 0:
        return lines

    metadata_by_id = {item["id"]: item for item in traffic_lights}
    ego_location = np.asarray(ego_pose[:3], dtype=np.float64)
    candidates = [
        (float(np.linalg.norm(np.asarray(metadata_by_id[item["id"]]["trigger_location"])
                              - ego_location)), item["id"], state_name)
        for item, state_name in zip(traffic_light_states, normalized)
        if item["id"] in metadata_by_id
    ]
    nearest = sorted(candidates, key=lambda item: (item[0], item[1]))[:nearest_count]
    return lines + ["TL #{} {} {:.1f}m".format(light_id, state_name.upper(), distance)
                    for distance, light_id, state_name in nearest]


def _relevant_control_hud_line(control):
    """把新旧数据来源与当前相关灯摘要成一行英文，兼容 OpenCV 内置字体。"""
    if control is None:
        return "Relevant TL [legacy]: field unavailable"
    if not control.get("valid"):
        return "Relevant TL [native]: none"
    return "Relevant TL [native]: #{} {} {:.1f}m".format(
        control["id"], str(control.get("state", "unknown")).upper(),
        float(control["route_distance"]))


# 模态开关在 HUD 中的展示顺序：(available 键, 提示标签, state 键)，仅列出本场景实际存在的模态
_TOGGLE_ITEMS = (
    ("rgb", "[r]rgb", "show_rgb"),
    ("depth", "[d]depth", "show_depth"),
    ("semantic", "[m]seg", "show_semantic"),
    ("optical_flow", "[f]flow", "show_flow"),
    ("lidar", "[v]bev", "show_bev"),
)
_HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
_HUD_SCALE = 0.5
_HUD_LH = 20           # 行高


def _put_hud(canvas, fmeta, state, traffic_light_lines, avail):
    """左上角叠加帧号/播放态、主车状态、可用模态开关与交通灯信息；半透明底提升可读性。"""
    ego = fmeta["ego"]
    speed = float(np.linalg.norm(ego["velocity"])) * 3.6  # m/s -> km/h
    ctrl = ego["control"]
    play = ">> PLAY" if state.get("playing") else "|| PAUSE"
    lines = [
        "frame {}/{}  t={:.1f}s  {}".format(state["idx"], state["num_frames"] - 1,
                                            fmeta["sim_time"], play),
        "speed {:.1f} km/h   thr {:.2f}  steer {:+.2f}  brake {:.2f}".format(
            speed, ctrl["throttle"], ctrl["steer"], ctrl["brake"]),
        _toggle_line(state, avail),
    ] + traffic_light_lines
    if state.get("failed"):
        lines.insert(0, "FAILED: {}".format(state.get("failure_status") or "unknown"))

    text_width = max(cv2.getTextSize(t, _HUD_FONT, _HUD_SCALE, 1)[0][0] for t in lines)
    hud_w = min(canvas.shape[1] - 1, text_width + 16)
    hud_h = min(canvas.shape[0] - 1, _HUD_LH * len(lines) + 10)
    # 半透明黑底：先在副本上画实心矩形再与原图加权混合，避免遮死底下画面
    overlay = canvas[:hud_h, :hud_w].copy()
    overlay[:] = (0, 0, 0)
    cv2.addWeighted(overlay, 0.55, canvas[:hud_h, :hud_w], 0.45, 0, canvas[:hud_h, :hud_w])
    for i, text in enumerate(lines):
        cv2.putText(canvas, text, (8, _HUD_LH * i + 20), _HUD_FONT, _HUD_SCALE,
                    (240, 240, 240), 1, cv2.LINE_AA)


def _toggle_line(state, avail):
    """显示开关行：框/静态框常驻，其余仅列出本场景存在的相机/lidar 模态。"""
    parts = ["[b]box {}".format(_on(state["show_bbox"])),
             "[s]static {}".format(_on(state["show_static"]))]
    parts += ["{} {}".format(label, _on(state[skey]))
              for key, label, skey in _TOGGLE_ITEMS if avail.get(key)]
    return "  ".join(parts)


def _on(flag):
    return "ON" if flag else "off"


def _titled(img, text):
    """在面板左上角标注名称（带描边，深浅背景都可读）。"""
    out = img.copy()
    cv2.putText(out, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _hrow(panels):
    """同一行的相机面板间插竖直间隔条后横向拼接（面板等高，故可直接 hconcat）。"""
    if len(panels) == 1:
        return panels[0]
    sep = np.full((panels[0].shape[0], _GUTTER, 3), _BG, dtype=np.uint8)
    return cv2.hconcat([p for panel in panels for p in (sep, panel)][1:])


def _stack_rows(rows):
    """各模态行间插水平间隔条后纵向拼接（每行等宽：同相机数与同面板尺寸）。"""
    if len(rows) == 1:
        return rows[0]
    sep = np.full((_GUTTER, rows[0].shape[1], 3), _BG, dtype=np.uint8)
    return cv2.vconcat([r for row in rows for r in (sep, row)][1:])


def _join_lr(left, right):
    """左侧相机面板与右侧 BEV 间插竖直间隔条后拼接（right 已贴合 left 高度）。"""
    sep = np.full((left.shape[0], _GUTTER, 3), _BG, dtype=np.uint8)
    return cv2.hconcat([left, sep, right])


def _fit_height(img, height):
    """等比缩放到目标高度（用于把方形 BEV 贴合左侧面板高度）。"""
    scale = height / img.shape[0]
    return cv2.resize(img, (max(1, int(img.shape[1] * scale)), height))
