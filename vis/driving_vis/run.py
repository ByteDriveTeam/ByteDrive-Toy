"""驾驶可视化入口 CLI：逐帧渲染透视模态与 GT/预测三场、道路线、交通控制及轨迹并保存。

模块: vis/driving_vis/run.py
依赖: argparse, pathlib, cv2, numpy, torch, config.load_config, model.driving_model.DrivingModel,
      data.driving_dataset.DrivingDataset, data.driving_targets.BevParams, vis.driving_vis.render,
      vis.driving_vis.checks.run_checks
读取配置:
    driving_vis.checkpoint / scene / max_frames / save_dir / show_ground_truth / display_scale
    driving_vis.field_colormap / depth_colormap / depth_max_display_m / depth_min_display_m / depth_log
    driving_vis.lane_map.*（道路线类别配色与方向箭头样式）
    driving_vis.traffic_control.*（灯态配色、停止线阈值与轨迹叠加强度）
    data.dataset.dino_mean / dino_std（RGB 去归一化展示）
    data.driving.camera（读原始 Seg/Depth 展示）
    model.driving.bev.*（BEV 场几何与视场）/ fields.up_channels（场分辨率）
    model.driving.traffic_control.state_names（灯态显示名）
对外接口:
    - main(argv=None) -> None      # 命令行入口
说明: 复用 DrivingDataset 逐帧取模型输入与 GT 场/轨迹（同一编码路径，保证预测与 GT 口径一致），另经其 reader
      读同帧原始 Seg/Depth 展示。选定场景取前 max_frames 帧，每列一帧，行含 RGB/Seg/Depth（透视）与 GT/预测
      的风险/可行驶/分布场、带方向箭头的道路线图、停止线/灯态及叠加轨迹 BEV（俯视）。加载驾驶权重（strict=False，容忍缺失的冻结骨干键）；检查点
      不存在则随机初始化并告警，便于仅验证渲染管线。推理沿用模型内部 BF16/FP32 混精边界，渲染委托
      vis.driving_vis.render，结果按场景存 PNG。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from config import load_config
from data.driving_dataset import DrivingDataset
from data.driving_targets import BevParams
from model.driving_model import DrivingModel
from vis.driving_vis import render
from vis.driving_vis.checks.run_checks import check_scene_frames

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TILE_H = 224  # 合成画布统一行高（透视图与 BEV 图按此高等比缩放）


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _REPO_ROOT / p


def _resolve_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _load_weights(model: DrivingModel, checkpoint: str, device) -> None:
    """加载驾驶权重；检查点不存在则告警并保持随机初始化（仅验证渲染管线）。"""
    path = _resolve(checkpoint)
    if not path.is_file():
        print("[driving_vis] 检查点不存在: {}，使用随机初始化权重（仅验证渲染）。".format(path))
        return
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)  # 骨干键不在检查点内，故 strict=False
    print("[driving_vis] 已加载权重: {}（epoch={}）".format(path, ckpt.get("epoch", "?")))


def _select_frames(dataset: DrivingDataset, scene: str, max_frames: int):
    """按场景筛选帧的数据集索引；scene 为空则取第一个场景，取前 max_frames 帧（0=全部）。"""
    index = dataset.frame_index
    target = scene or index[0][0].name
    selected = [i for i, (scene_dir, _) in enumerate(index) if scene_dir.name == target]
    check_scene_frames(selected, target)
    return target, (selected if max_frames == 0 else selected[:max_frames])


def _bev_params(cfg) -> BevParams:
    """场分辨率 BEV 几何（与 field_decoder 输出/DrivingDataset 一致）。"""
    bev = cfg.model.driving.bev
    scale = 2 ** len(cfg.model.driving.fields.up_channels)
    return BevParams(bev.x_min_m, bev.x_max_m, bev.y_min_m, bev.y_max_m,
                     bev.height * scale, bev.width * scale)


def _predict_fields(outputs):
    """预测三场 logit → sigmoid 概率 numpy [H,W]。"""
    return {name: torch.sigmoid(outputs[name][0, 0]).cpu().numpy()
            for name in ("risk", "drivable", "distribution")}


def _predict_lane_map(outputs):
    """道路线类别 logits 与原始方向向量 → 类别图及单位有向切向量 numpy。"""
    lane_class = outputs["lane_class_logits"][0].argmax(dim=0).cpu().numpy()
    direction = outputs["lane_direction"][0]
    direction = direction / torch.linalg.vector_norm(direction, dim=0, keepdim=True).clamp_min(1e-12)
    return lane_class, direction.cpu().numpy()


def _predict_traffic_control(outputs, state_names, threshold):
    """把交通控制 logits 转成阈值化停止线、灯态类别图与峰值摘要。"""
    stop = torch.sigmoid(outputs["stop_line_logits"][0, 0]).cpu().numpy()
    state_prob = torch.softmax(outputs["traffic_light_state_logits"][0], dim=0).cpu().numpy()
    state_map = state_prob.argmax(axis=0)
    peak = np.unravel_index(int(stop.argmax()), stop.shape)
    line_score = float(stop[peak])
    active = line_score > threshold
    state_id = int(state_map[peak])
    state_score = float(state_prob[state_id][peak])
    annotations = (["state {}".format(state_names[state_id]),
                    "line {:.2f}  state {:.2f}".format(line_score, state_score)]
                   if active else ["state none", "line {:.2f}".format(line_score)])
    return np.where(stop > threshold, stop, 0.0), state_map, annotations


def _ground_truth_traffic_control(sample, state_names):
    """提取停止线/灯态真值，并生成灯态、停车要求和距离摘要。"""
    stop = sample["stop_line"].numpy()
    state_map = sample["traffic_light_state"].numpy()
    state_valid = sample["traffic_light_state_valid"].numpy()
    if not bool((stop > 0).any()):
        return stop, state_map, state_valid, ["state none"]
    valid_pixels = state_valid > 0
    state = state_names[int(state_map[valid_pixels][0])] if bool(valid_pixels.any()) else "unknown"
    must_stop = bool(float(sample["red_stop_valid"]) > 0)
    distance = float(sample["stop_distance"])
    annotations = ["state {}".format(state),
                   "must_stop {}  d {:.1f}m".format("yes" if must_stop else "no", distance)]
    return stop, state_map, state_valid, annotations


def main(argv=None) -> None:
    """驾驶可视化主流程。"""
    parser = argparse.ArgumentParser(description="ByteDrive 驾驶模型可视化")
    parser.add_argument("--config", default=None, help="主配置文件路径（缺省用 config/default.yaml）")
    parser.add_argument("--env", default=None, help="环境覆盖名（叠加 config/<env>.yaml）")
    parser.add_argument("--checkpoint", default=None, help="覆盖 driving_vis.checkpoint 的权重路径")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.env)
    dv = cfg.driving_vis
    device = _resolve_device()
    mean = np.asarray(cfg.data.dataset.dino_mean, dtype=np.float32)
    std = np.asarray(cfg.data.dataset.dino_std, dtype=np.float32)
    camera = cfg.data.driving.camera
    fov = cfg.model.driving.bev.fov_deg
    bev = _bev_params(cfg)

    model = DrivingModel(cfg).to(device).eval()
    _load_weights(model, args.checkpoint or dv.checkpoint, device)
    dataset = DrivingDataset(cfg)
    scene, indices = _select_frames(dataset, dv.scene, dv.max_frames)

    panels = {name: [] for name in _ROW_ORDER}
    for idx in indices:
        _accumulate_frame(dataset, idx, model, device, cfg, dv, camera, bev, fov, mean, std, panels)

    rows = [(label, panels[label]) for label in _ROW_ORDER
            if panels[label] and (dv.show_ground_truth or not label.startswith("gt "))]
    canvas = render.compose_canvas(rows, _TILE_H)
    if dv.display_scale != 1.0:
        canvas = cv2.resize(canvas, None, fx=dv.display_scale, fy=dv.display_scale)

    save_dir = _resolve(dv.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "{}_n{:02d}.png".format(scene, len(indices))
    cv2.imwrite(str(out_path), canvas)
    print("[driving_vis] 已保存 {}".format(out_path))


# 行顺序（透视 → BEV 三场/道路线/交通控制 GT/预测 → 轨迹 BEV）；gt 行可按配置跳过
_ROW_ORDER = ("rgb", "seg", "depth",
              "gt risk", "pred risk", "gt drivable", "pred drivable",
              "gt dist", "pred dist", "gt lanes", "pred lanes",
              "gt traffic", "pred traffic", "gt traj", "pred traj")


def _accumulate_frame(dataset, idx, model, device, cfg, dv, camera, bev, fov, mean, std, panels):
    """对单帧推理并把各模态面板追加进 panels（每列一帧）。"""
    sample = dataset[idx]
    scene_dir, frame_idx = dataset.frame_index[idx]
    frame = dataset.reader(scene_dir).frame(frame_idx)

    with torch.no_grad():
        outputs = model(sample["rgb"].unsqueeze(0).to(device), sample["intrinsics"].unsqueeze(0).to(device),
                        sample["extrinsics"].unsqueeze(0).to(device),
                        sample["target_point"].unsqueeze(0).to(device),
                        sample["previous_rgb"].unsqueeze(0).to(device),
                        sample["previous_to_current"].unsqueeze(0).to(device),
                        sample["previous_valid"].unsqueeze(0).to(device))
    pred = _predict_fields(outputs)
    inview = sample["inview"].numpy()
    gt = {k: sample[k].numpy() for k in ("risk", "drivable", "distribution")}
    _append_perspective_panels(sample, frame, camera, dv, mean, std, panels)
    _append_field_panels(gt, pred, inview, dv.field_colormap, panels)
    _append_lane_panels(sample, outputs, inview, dv.lane_map, panels)
    traffic_layers = _append_traffic_panels(sample, outputs, cfg, dv, inview, panels)
    _append_trajectory_panels(sample, outputs, gt, pred, traffic_layers, dv, bev, fov, panels)


def _append_perspective_panels(sample, frame, camera, dv, mean, std, panels):
    """追加 RGB、语义和深度透视面板。"""
    rgb = sample["rgb"].cpu().numpy() * std[:, None, None] + mean[:, None, None]
    panels["rgb"].append(render.to_display_bgr(rgb))
    panels["seg"].append(render.colorize_semantic(np.ascontiguousarray(frame["semantic"][camera])))
    panels["depth"].append(render.colorize_depth(
        np.ascontiguousarray(frame["depth"][camera]).astype(np.float32),
        dv.depth_colormap, dv.depth_max_display_m, dv.depth_min_display_m, dv.depth_log))


def _append_field_panels(gt, pred, inview, colormap, panels):
    """追加三场的真值与预测面板。"""
    for name in ("risk", "drivable", "distribution"):
        key = "dist" if name == "distribution" else name
        panels["gt " + key].append(render.colorize_field(gt[name], colormap, inview))
        panels["pred " + key].append(render.colorize_field(pred[name], colormap, inview))


def _append_lane_panels(sample, outputs, inview, lane_vis, panels):
    """追加带方向箭头的道路线真值与预测面板。"""
    pred_class, pred_direction = _predict_lane_map(outputs)
    lane_args = (lane_vis.class_colors, lane_vis.arrow_color, lane_vis.arrow_stride_px,
                 lane_vis.arrow_length_px, lane_vis.arrow_thickness, lane_vis.arrow_tip_ratio, inview)
    panels["gt lanes"].append(render.colorize_lane_map(
        sample["lane_class"].numpy(), sample["lane_direction"].numpy(), *lane_args))
    panels["pred lanes"].append(render.colorize_lane_map(
        pred_class, pred_direction, *lane_args))


def _append_traffic_panels(sample, outputs, cfg, dv, inview, panels):
    """追加停止线/灯态面板，并返回供轨迹 BEV 叠加的图层。"""
    state_names = cfg.model.driving.traffic_control.state_names
    traffic_vis = dv.traffic_control
    pred_stop, pred_state, pred_notes = _predict_traffic_control(
        outputs, state_names, traffic_vis.line_threshold)
    gt_stop, gt_state, gt_valid, gt_notes = _ground_truth_traffic_control(sample, state_names)
    traffic_args = (traffic_vis.state_colors, traffic_vis.unknown_color, inview)
    gt_traffic = render.colorize_traffic_control(
        gt_stop, gt_state, gt_valid, *traffic_args, gt_notes)
    pred_traffic = render.colorize_traffic_control(
        pred_stop, pred_state, None, *traffic_args, pred_notes)
    panels["gt traffic"].append(gt_traffic)
    panels["pred traffic"].append(pred_traffic)
    return gt_stop, gt_traffic, pred_stop, pred_traffic


def _append_trajectory_panels(sample, outputs, gt, pred, traffic_layers, dv, bev, fov, panels):
    """在三场底图叠加停止线，再追加真值与多模态预测轨迹。"""
    gt_stop, gt_traffic, pred_stop, pred_traffic = traffic_layers
    inview = sample["inview"].numpy()
    gt_base = render.bev_scene_composite(gt["risk"], gt["drivable"], gt["distribution"], inview)
    pred_base = render.bev_scene_composite(pred["risk"], pred["drivable"], pred["distribution"], inview)
    gt_base = render.overlay_traffic_control(
        gt_base, gt_traffic, gt_stop > 0, dv.traffic_control.overlay_alpha)
    pred_base = render.overlay_traffic_control(
        pred_base, pred_traffic, pred_stop > 0, dv.traffic_control.overlay_alpha)
    gt_traj, gt_valid = sample["trajectory"].numpy(), sample["traj_valid"].numpy()
    empty_modes = np.zeros((0, gt_traj.shape[0], 2), dtype=np.float32)  # gt 面板只画 GT
    panels["gt traj"].append(render.draw_trajectories(
        gt_base, empty_modes, np.zeros(0), gt_traj, gt_valid, bev, fov, draw_gt=True))
    trajectories = outputs["trajectories"][0].cpu().numpy()
    confidence = outputs["confidence"][0].cpu().numpy()
    panels["pred traj"].append(render.draw_trajectories(
        pred_base, trajectories, confidence, gt_traj, gt_valid, bev, fov, draw_gt=dv.show_ground_truth))


if __name__ == "__main__":
    main()
