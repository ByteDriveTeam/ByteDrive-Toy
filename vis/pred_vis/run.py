"""预测可视化入口 CLI：加载配置与权重 → 对场景逐窗推理 → 渲染三头预测与 GT 对照并保存。

模块: vis/pred_vis/run.py
依赖: argparse, pathlib, cv2, numpy, torch, config.load_config,
      model.perception_model.PerceptionModel, data.perception_dataset.PerceptionDataset,
      data.target_encoding.physics_decode, vis.pred_vis.render, vis.pred_vis.checks.run_checks
读取配置:
    pred_vis.checkpoint / scene / max_windows / save_dir / show_ground_truth
    pred_vis.display_scale / depth_colormap / depth_max_display_m / depth_min_display_m
    pred_vis.depth_log / flow_max_display
    data.dataset.dino_mean / dino_std（RGB 去归一化展示）
    model.physics.symlog_scale（预测/GT 由 Symlog 空间解码回物理量）
    model.physics.depth_max_m（深度按范围二分类掩码：超范围像素置此值平铺展示）
对外接口:
    - main(argv=None) -> None      # 命令行入口
说明: 复用 PerceptionDataset 枚举窗口并取归一化 RGB 与 GT 目标（同一编码/解码路径，保证预测与 GT 物理口径
      一致）。加载可训练权重（strict=False，容忍缺失的冻结骨干键）；检查点不存在则随机初始化并告警，便于
      仅验证渲染管线。推理沿用模型内部 BF16/FP32 混精边界。渲染委托 vis.pred_vis.render，结果按窗口存 PNG。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from config import load_config
from data.perception_dataset import PerceptionDataset
from data.target_encoding import physics_decode
from model.perception_model import PerceptionModel
from vis.pred_vis import render
from vis.pred_vis.checks.run_checks import check_scene_windows

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _REPO_ROOT / p


def _resolve_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _load_weights(model: PerceptionModel, checkpoint: str, device) -> None:
    """加载可训练权重；检查点不存在则告警并保持随机初始化（仅验证渲染管线）。"""
    path = _resolve(checkpoint)
    if not path.is_file():
        print("[pred_vis] 检查点不存在: {}，使用随机初始化权重（仅验证渲染）。".format(path))
        return
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model", ckpt)  # 兼容纯 state_dict 或 {epoch,model,optimizer}
    model.load_state_dict(state, strict=False)  # 骨干键不在检查点内，故 strict=False
    print("[pred_vis] 已加载权重: {}（epoch={}）".format(path, ckpt.get("epoch", "?")))


def _select_windows(dataset: PerceptionDataset, scene: str, max_windows: int):
    """按场景筛选窗口的 (数据集索引, 起始帧)；scene 为空则取第一个场景。"""
    index = dataset.window_index
    target = scene or index[0][0].name
    selected = [(i, start) for i, (scene_dir, start) in enumerate(index) if scene_dir.name == target]
    check_scene_windows(selected, target)
    return target, (selected if max_windows == 0 else selected[:max_windows])


def _decode_scalar_map(symlog_map: torch.Tensor, scale: float) -> np.ndarray:
    """把 [T,H,W] 的 scale·symlog 值解码回物理量并转 numpy。"""
    return physics_decode(symlog_map, scale).cpu().numpy()


def _build_rows(rgb: torch.Tensor, pred, gt, pv, mean, std, scale: float):
    """构造 render_grid 的行：RGB + 预测三模态（+ 可选 GT 三模态）。逐帧着色。"""
    frames = rgb.shape[0]
    rgb01 = (rgb.cpu().numpy() * std[:, None, None] + mean[:, None, None])  # 去归一化
    rows = [("rgb", [render.to_display_bgr(rgb01[t]) for t in range(frames)])]
    rows += _modality_rows("pred", pred, pv)
    if gt is not None:
        rows += _modality_rows("gt", gt, pv)
    return rows


def _modality_rows(tag: str, maps, pv):
    """一组来源（pred/gt）的语义/深度/光流三行面板。"""
    sem, depth_m, velocity = maps["semantic"], maps["depth"], maps["flow"]
    frames = sem.shape[0]
    return [
        ("{} seg".format(tag), [render.colorize_semantic(sem[t]) for t in range(frames)]),
        ("{} depth".format(tag),
         [render.colorize_depth(depth_m[t], pv.depth_colormap, pv.depth_max_display_m,
                                pv.depth_min_display_m, pv.depth_log)
          for t in range(frames)]),
        ("{} flow".format(tag),
         [render.colorize_flow(velocity[:, t], pv.flow_max_display) for t in range(frames)]),
    ]


def _predictions(outputs, scale: float, depth_max_m: float):
    """把模型输出解码为可着色的物理量（语义标签 / 深度米 / 速度）。

    深度仅在预测「范围内」的像素展示回归值：以深度头 ch1（范围二分类 logit>0 ⇔ sigmoid>0.5）
    为掩码，超范围像素直接置 depth_max_m（如 128m），避免未受监督的超范围回归值污染显示。
    """
    semantic = outputs["semantic"][0].argmax(dim=0).cpu().numpy()          # [T,H,W]
    depth = _decode_scalar_map(outputs["depth"][0, 0], scale)              # [T,H,W]
    in_range = (outputs["depth"][0, 1] > 0.0).cpu().numpy()               # 预测范围内掩码
    depth = np.where(in_range, depth, depth_max_m)
    velocity = physics_decode(outputs["flow"][0], scale).cpu().numpy()     # [2,T,H,W]
    return {"semantic": semantic, "depth": depth, "flow": velocity}


def _ground_truth(sample, scale: float, depth_max_m: float):
    """把数据集 GT 目标解码为与预测同口径的物理量。

    深度同预测口径：GT 超范围像素（depth_inrange==0）直接置 depth_max_m，使 pred/gt 两行可直接对照。
    """
    semantic = sample["semantic"].numpy()                                  # [T,H,W]
    depth = _decode_scalar_map(sample["depth_target"], scale)              # [T,H,W]
    in_range = sample["depth_inrange"].numpy() > 0.5                        # GT 范围内掩码
    depth = np.where(in_range, depth, depth_max_m)
    velocity = physics_decode(sample["flow_target"], scale).numpy()        # [2,T,H,W]
    return {"semantic": semantic, "depth": depth, "flow": velocity}


def main(argv=None) -> None:
    """预测可视化主流程。"""
    parser = argparse.ArgumentParser(description="ByteDrive 感知模型预测可视化")
    parser.add_argument("--config", default=None, help="主配置文件路径（缺省用 config/default.yaml）")
    parser.add_argument("--env", default=None, help="环境覆盖名（叠加 config/<env>.yaml）")
    parser.add_argument("--checkpoint", default=None, help="覆盖 pred_vis.checkpoint 的权重路径")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.env)
    pv = cfg.pred_vis
    device = _resolve_device()
    scale = cfg.model.physics.symlog_scale
    depth_max_m = cfg.model.physics.depth_max_m  # 超范围像素按此值(如128m)平铺展示
    mean = np.asarray(cfg.data.dataset.dino_mean, dtype=np.float32)
    std = np.asarray(cfg.data.dataset.dino_std, dtype=np.float32)

    model = PerceptionModel(cfg).to(device).eval()
    _load_weights(model, args.checkpoint or pv.checkpoint, device)
    dataset = PerceptionDataset(cfg)
    scene, windows = _select_windows(dataset, pv.scene, pv.max_windows)

    save_dir = _resolve(pv.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for dataset_index, start in windows:
        sample = dataset[dataset_index]
        with torch.no_grad():
            outputs = model(sample["rgb"].unsqueeze(0).to(device))
        gt = _ground_truth(sample, scale, depth_max_m) if pv.show_ground_truth else None
        rows = _build_rows(sample["rgb"], _predictions(outputs, scale, depth_max_m),
                           gt, pv, mean, std, scale)
        canvas = render.render_grid(rows)
        if pv.display_scale != 1.0:
            canvas = cv2.resize(canvas, None, fx=pv.display_scale, fy=pv.display_scale)
        out_path = save_dir / "{}_w{:04d}.png".format(scene, start)
        cv2.imwrite(str(out_path), canvas)
        print("[pred_vis] 已保存 {}".format(out_path))


if __name__ == "__main__":
    main()
