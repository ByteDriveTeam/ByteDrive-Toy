"""按命令行模式保存 BEVSeg 数据集或压缩器重建与量化特征 PCA 可视化。

模块: vis/bevseg_vis/run.py
依赖: config, torch, data.bevseg_synthesis, model.bevseg_compressor,
      vis.bevseg_vis, vis.bevseg_vis.checks.run_checks, vis.data_vis.reader
读取配置: bevseg_vis.inference/checkpoint/scene/frame/save_dir/device/semantic_threshold,
          data.bevseg
对外接口:
    - main(argv=None) -> None
说明: failed 仅代表驾驶结果；关闭推理时不构建模型，开启推理但无权重时明确告警并使用随机权重。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import load_config
from data.bevseg_synthesis import BevSegRasterizer
from model.bevseg_compressor import BEVSegCompressor
from vis.bevseg_vis import save_bevseg_history, save_bevseg_reconstruction
from vis.bevseg_vis.checks.run_checks import check_frame
from vis.data_vis.reader import SceneReader, list_scenes


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else _REPO_ROOT / path


def _resolve_device(requested):
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        print("[bevseg-vis] CUDA 不可用，回退 CPU")
        return torch.device("cpu")
    return torch.device(requested)


def _pick_scene(value, root):
    scenes = list_scenes(root)
    if not scenes:
        raise FileNotFoundError("没有找到真实场景目录: {}".format(root))
    if not value:
        return scenes[0]
    candidate = _resolve(value)
    if candidate.is_dir():
        return candidate
    by_name = root / str(value)
    if by_name.is_dir():
        return by_name
    if str(value).isdigit() and 0 <= int(value) < len(scenes):
        return scenes[int(value)]
    raise ValueError("无法解析场景: {}".format(value))


def _load_checkpoint(model, checkpoint, device):
    path = _resolve(checkpoint)
    if not path.is_file():
        print("[bevseg-vis][WARNING] 推理已启用，但检查点不存在: {}".format(path))
        print("[bevseg-vis][WARNING] 将使用随机初始化权重；"
              "输出仅验证推理/渲染链路，不代表重建质量。")
        return "random"
    payload = torch.load(path, map_location=device, weights_only=True)
    state = payload.get("model", payload)
    model.load_state_dict(state, strict=True)
    epoch = payload.get("epoch", "?") if isinstance(payload, dict) else "?"
    print("[bevseg-vis] 已加载权重: {}（epoch={}）".format(path, epoch))
    return epoch


def _rasterize_frame(reader, rasterizer, frame):
    """栅格化一个当前 ego 坐标系下的单帧样本。"""
    current_pose = reader.frame_meta(frame)["ego"]["transform"]
    map_obj = rasterizer.load_map(reader.meta)
    return rasterizer.rasterize_frame(reader.meta, reader.frame_meta(frame), map_obj, current_pose)


def _model_input(sample):
    array = np.concatenate((sample["semantic"], sample["direction"]), axis=0)
    return torch.from_numpy(np.ascontiguousarray(array)).float().unsqueeze(0)


def main(argv=None):
    """执行真实单帧数据可视化，并按需执行压缩器推理。"""
    parser = argparse.ArgumentParser(description="BEVSeg 数据集与压缩器重建/PCA 可视化")
    parser.add_argument("--config", default=None, help="主配置文件路径")
    parser.add_argument("--env", default=None, help="环境覆盖名")
    inference_group = parser.add_mutually_exclusive_group()
    inference_group.add_argument("--inference", dest="inference", action="store_true",
                                 help="启用压缩器推理，显示重建与 PCA")
    inference_group.add_argument("--no-inference", dest="inference", action="store_false",
                                 help="禁用压缩器推理，仅显示数据集")
    parser.set_defaults(inference=None)
    parser.add_argument("--checkpoint", default=None,
                        help="覆盖 bevseg_vis.checkpoint 的检查点路径")
    parser.add_argument("--scene", default=None,
                        help="覆盖 bevseg_vis.scene：scene_XXXXXX、目录路径或场景序号")
    parser.add_argument("--frame", type=int, default=None,
                        help="覆盖 bevseg_vis.frame：当前帧号，-1 表示最后一帧")
    parser.add_argument("--output", default=None,
                        help="覆盖自动输出路径的 PNG 文件路径")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.env)
    vis_cfg = cfg.bevseg_vis
    data_cfg = cfg.data.bevseg
    inference = vis_cfg.inference if args.inference is None else args.inference

    scene_root = _resolve(data_cfg.scene_root)
    scene_dir = _pick_scene(args.scene if args.scene is not None else vis_cfg.scene, scene_root)
    reader = SceneReader(scene_dir)
    try:
        requested_frame = vis_cfg.frame if args.frame is None else args.frame
        frame = reader.num_frames - 1 if requested_frame == -1 else requested_frame
        check_frame(frame, reader.num_frames, scene_dir)
        rasterizer = BevSegRasterizer(data_cfg)
        sample = _rasterize_frame(reader, rasterizer, frame)
        suffix = "reconstruction_pca" if inference else "dataset"
        output = (_resolve(args.output) if args.output else
                  _resolve(vis_cfg.save_dir) /
                  "bevseg_{}_f{:06d}_{}.png".format(scene_dir.name, frame, suffix))
        if inference:
            device = _resolve_device(vis_cfg.device)
            model = BEVSegCompressor(cfg).to(device).eval()
            epoch = _load_checkpoint(model, args.checkpoint or vis_cfg.checkpoint, device)
            with torch.inference_mode():
                outputs = model(_model_input(sample).to(device), epoch=None, sample=False)
            logits = outputs["reconstruction_logits"][0].reshape(
                len(rasterizer.layers) + 2,
                data_cfg.resolution, data_cfg.resolution).float().cpu().numpy()
            codes = outputs["codes"][0].float().cpu().numpy()
            save_bevseg_reconstruction(
                [sample], logits[None], codes, rasterizer.layers, vis_cfg.semantic_threshold,
                output, frame_ids=[frame],
                metadata={"scene": scene_dir.name, "frame": frame, "epoch": epoch})
        else:
            print("[bevseg-vis] 模式=仅数据集（未构建模型、未执行推理）")
            save_bevseg_history(
                [sample], rasterizer.layers, output, frame_ids=[frame])
        print("[bevseg-vis] mode={} scene={} frame={} output={}".format(
            "inference" if inference else "dataset", scene_dir.name, frame, output))
    finally:
        reader.close()


if __name__ == "__main__":
    main()
