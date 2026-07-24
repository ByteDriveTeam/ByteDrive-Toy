"""加载驾驶权重、维护双帧状态，并按置信度/安全场/路线一致性选择闭环轨迹。

模块: clone_loop/inference/inference.py
依赖: collections, pathlib, numpy, torch, torch.nn.functional, model.driving_model,
      clone_loop.inference.checks.inference_checks
读取配置:
    clone_loop.inference.checkpoint / device / min_weight_coverage / confidence_weight /
        risk_weight / drivable_weight / route_alignment_weight / max_abs_waypoint_m
    clone_loop.camera.width / height
    clone_loop.control.waypoint_dt_s / clone_loop.simulation.fixed_delta_seconds
    data.dataset.dino_mean / dino_std
    model.driving.bev.x_min_m / x_max_m / y_min_m / y_max_m
    （驾驶模型构造继续读取 config.model）
对外接口:
    - ClosedLoopPolicy(cfg)
        .reset() -> None
        .infer(frame_bgr, observation) -> dict
说明: 历史帧和位姿只在主环境保存；每个 episode 首帧以当前帧回填并把 previous_valid 置零。
"""

from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from clone_loop.inference.checks.inference_checks import (
    check_frame,
    check_observation,
    check_trajectory_candidates,
)
from model.driving_model import DrivingModel


__all__ = ["ClosedLoopPolicy"]

_REPO_ROOT = Path(__file__).resolve().parents[2]


class ClosedLoopPolicy:
    """驾驶模型在线推理器与安全感知轨迹选择器。"""

    def __init__(self, cfg):
        self._cfg = cfg
        self._inference_cfg = cfg.clone_loop.inference
        self._camera_cfg = cfg.clone_loop.camera
        self._device = self._resolve_device(self._inference_cfg.device)
        self._mean = torch.tensor(
            cfg.data.dataset.dino_mean, dtype=torch.float32).view(3, 1, 1)
        self._std = torch.tensor(
            cfg.data.dataset.dino_std, dtype=torch.float32).view(3, 1, 1)
        self._history_steps = max(int(round(
            cfg.clone_loop.control.waypoint_dt_s
            / cfg.clone_loop.simulation.fixed_delta_seconds)), 1)
        self._model = DrivingModel(cfg).to(self._device).eval()
        self._load_weights(self._inference_cfg.checkpoint)
        self.reset()

    @staticmethod
    def _resolve_device(requested):
        """请求 CUDA 但不可用时稳定回退 CPU。"""
        if str(requested).startswith("cuda") and not torch.cuda.is_available():
            print("[clone_loop] CUDA 不可用，推理回退 CPU")
            return torch.device("cpu")
        return torch.device(requested)

    def _load_weights(self, checkpoint):
        """只加载当前模型存在且形状兼容的非骨干权重，并检查覆盖率。"""
        path = Path(checkpoint)
        path = path if path.is_absolute() else _REPO_ROOT / path
        if not path.is_file():
            raise FileNotFoundError("闭环驾驶检查点不存在: {}".format(path))
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        saved = payload.get("model", payload)
        current = self._model.state_dict()
        expected = {name for name in current if "backbone." not in name}
        compatible = {
            name: value for name, value in saved.items()
            if name in current and tuple(value.shape) == tuple(current[name].shape)
        }
        coverage = len(expected.intersection(compatible)) / max(len(expected), 1)
        if coverage < self._inference_cfg.min_weight_coverage:
            raise RuntimeError(
                "闭环权重覆盖率 {:.1%} 低于要求 {:.1%}".format(
                    coverage, self._inference_cfg.min_weight_coverage))
        self._model.load_state_dict(compatible, strict=False)
        print("[clone_loop] 已加载驾驶权重 {}（epoch={}，覆盖率 {:.1%}）".format(
            path, payload.get("epoch", "?"), coverage))

    def reset(self):
        """清空上一帧图像与位姿，隔离不同 episode 的时序状态。"""
        self._history = deque(maxlen=self._history_steps)

    def infer(self, frame_bgr, observation):
        """对当前共享帧推理，返回选中轨迹、行为概率和各模态评分。"""
        check_frame(frame_bgr, self._camera_cfg.height, self._camera_cfg.width)
        check_observation(observation)
        current_rgb = self._normalize(frame_bgr)
        current_pose = np.asarray(observation["pose"], dtype=np.float64)
        history_ready = len(self._history) >= self._history_steps
        previous_rgb, previous_pose = (
            self._history[-self._history_steps]
            if history_ready else (current_rgb, current_pose))
        previous_to_current = (
            _planar_previous_to_current(previous_pose, current_pose)
            if history_ready else np.eye(3, dtype=np.float32))
        previous_valid = float(history_ready)

        inputs = self._inputs(
            current_rgb, previous_rgb, previous_to_current, previous_valid, observation)
        with torch.inference_mode():
            outputs = self._model(**inputs)
            selected, scores = self._select(outputs, inputs["target_point"])
            behavior = torch.sigmoid(outputs["behavior_logits"][0])

        self._history.append((current_rgb.detach(), current_pose.copy()))
        return {
            "trajectory": outputs["trajectories"][0, selected].float().cpu().numpy(),
            "behavior_probabilities": behavior.float().cpu().numpy(),
            "mode": int(selected),
            "mode_scores": scores.float().cpu().numpy(),
            "confidence": outputs["confidence"][0].float().cpu().numpy(),
        }

    def _normalize(self, bgr):
        """BGR uint8 → DINO 所需的归一化 RGB `[1,3,H,W]`。"""
        rgb = torch.from_numpy(np.ascontiguousarray(bgr[:, :, ::-1])).float() / 255.0
        return ((rgb.permute(2, 0, 1) - self._mean) / self._std).unsqueeze(0)

    def _inputs(self, current, previous, transform, previous_valid, observation):
        """把纯数值观测装配为 DrivingModel 的命名张量输入。"""
        tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=self._device)
        return {
            "rgb": current.to(self._device),
            "intrinsics": tensor(observation["intrinsics"]).unsqueeze(0),
            "extrinsics": tensor(observation["extrinsics"]).unsqueeze(0),
            "target_point": tensor(observation["target_point"]).unsqueeze(0),
            "ego_velocity": tensor(observation["ego_velocity"]).unsqueeze(0),
            "previous_rgb": previous.to(self._device),
            "previous_to_current": tensor(transform).unsqueeze(0),
            "previous_valid": tensor([previous_valid]),
        }

    def _select(self, outputs, target_point):
        """沿每条候选轨迹采样预测场，并与导航目标方向共同重排模型置信度。"""
        trajectories = outputs["trajectories"][0].float()
        check_trajectory_candidates(trajectories, self._inference_cfg.max_abs_waypoint_m)
        cfg = self._inference_cfg
        risk = self._sample_field(
            torch.sigmoid(outputs["risk"].float()), trajectories, outside_value=1.0)
        drivable = self._sample_field(
            torch.sigmoid(outputs["drivable"].float()), trajectories, outside_value=0.0)
        target = F.normalize(target_point[0].float(), dim=0)
        endpoints = F.normalize(trajectories[:, -1], dim=-1)
        alignment_cost = 1.0 - torch.sum(endpoints * target[None], dim=-1)
        scores = (
            cfg.confidence_weight * outputs["confidence"][0].float()
            - cfg.risk_weight * risk
            - cfg.drivable_weight * (1.0 - drivable)
            - cfg.route_alignment_weight * alignment_cost
        )
        return int(torch.argmax(scores).item()), scores

    def _sample_field(self, field, trajectories, outside_value):
        """把 ego xy 转成 grid_sample 坐标，越界按调用方指定的最坏场值计入。"""
        bev = self._cfg.model.driving.bev
        x, y = trajectories[..., 0], trajectories[..., 1]
        grid_x = 2.0 * (y - bev.y_min_m) / (bev.y_max_m - bev.y_min_m) - 1.0
        grid_y = 1.0 - 2.0 * (x - bev.x_min_m) / (bev.x_max_m - bev.x_min_m)
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
        sampled = F.grid_sample(
            field, grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0, 0]
        valid = (grid_x.abs() <= 1.0) & (grid_y.abs() <= 1.0)
        safe = torch.where(valid, sampled, torch.full_like(sampled, outside_value))
        return safe.mean(dim=-1)


def _planar_previous_to_current(previous_pose, current_pose):
    """由两帧 CARLA 平面位姿构造上一帧 ego xy 到当前 ego xy 的齐次刚体变换。"""
    previous = _planar_pose_matrix(previous_pose)
    current = _planar_pose_matrix(current_pose)
    return (np.linalg.inv(current) @ previous).astype(np.float32)


def _planar_pose_matrix(pose):
    """CARLA 位姿 `[x,y,z,roll,pitch,yaw]` 的平面齐次矩阵。"""
    yaw = np.radians(float(pose[5]))
    cosine, sine = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cosine, -sine, pose[0]],
        [sine, cosine, pose[1]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
