"""驾驶训练切片缓存。
模块: data/driving_cache/driving_cache.py
读取配置: data.driving_cache.*、trajectory_vocabulary.output_dir/action_output、model.unified_driving.waypoint_dt_s/nearest_weight_floor/nearest_decay_s/annotation_*/navigation_commands
说明: 每个三秒切片只保存 Top1 词索引及其标准化二维加速度残差目标；BEV latent 在线生成。
对外接口: DrivingSliceCache、DrivingSliceDataset、pack_driving_sample、prepare_driving_cache。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch.utils.data import Dataset

from data.carla_data_collector.common.navigation_command import (
    navigation_command_from_route as _route_command,
)
from data.driving_cache.checks.driving_cache_checks import check_navigation_route
from data.driving_cache.trajectory_risk import SceneTrajectoryRisk
from data.navigation_goal_filter import (
    TRAINING_SELECTION, is_navigation_goal_approach, navigation_goal_cutoff,
    window_before_navigation_goal,
)
from data.single_frame_base import resolve_repo_path, successful_scene_meta
from model.driving_model import nearest_planar_action_words
from model.driving_model.unified_driving_model import signed_log


__all__ = ["DrivingSliceCache", "DrivingSliceDataset", "trajectory_to_planar_accelerations",
           "standardize_actions", "unstandardize_actions", "pack_driving_sample",
           "navigation_command_from_route", "slice_driving_samples", "prepare_driving_cache"]


class DrivingSliceCache:
    """版本隔离、原子写入的 3 秒驾驶样本缓存。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.root = resolve_repo_path(cfg.data.driving_cache.dir)
        self.namespace = self._namespace(cfg.data.driving_cache.version, cfg)
        self.path = self.root / self.namespace
        self.path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _namespace(version, cfg):
        artifact = resolve_repo_path(
            str(Path(cfg.trajectory_vocabulary.output_dir) / cfg.trajectory_vocabulary.action_output))
        artifact_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()[:16] if artifact.is_file() else None
        payload = {"version": version, "slice_duration_s": float(cfg.data.driving_cache.slice_duration_s),
                   "training_selection": TRAINING_SELECTION,
                   "trajectory_target": "top1_cartesian_word_and_standardized_physical_residual_v1",
                   "action_vocabulary": cfg.trajectory_vocabulary.action_output,
                   "action_vocabulary_hash": artifact_hash,
                   "action_vocabulary_size": cfg.model.unified_driving.action_vocab_size,
                   "navigation_commands": cfg.model.unified_driving.navigation_commands,
                   "recovery_commands": cfg.model.unified_driving.recovery_commands,
                   "navigation_lookahead_m": cfg.data.driving_cache.navigation_lookahead_m,
                   "navigation_turn_threshold_deg": cfg.data.driving_cache.navigation_turn_threshold_deg,
                   "navigation_lane_match_radius_m": cfg.data.driving_cache.navigation_lane_match_radius_m,
                   "navigation_heading_tolerance_deg": cfg.data.driving_cache.navigation_heading_tolerance_deg,
                   "navigation_lane_stable_points": cfg.data.driving_cache.navigation_lane_stable_points,
                   "annotation_schema_version": 8,
                   "risk_targets": "masked_conflict_and_lane_availability",
                   "collision_targets": "solid_static_and_dynamic_height_aware_v2",
                   "frames": 4, "waypoints": 30}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        return "{}-{}".format(version, digest)

    def _file(self, key):
        safe_key = hashlib.sha256(str(key).encode("utf-8")).hexdigest()
        return self.path / (safe_key + ".pt")

    def load(self, key):
        path = self._file(key)
        if not path.is_file():
            return None
        try:
            value = torch.load(path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    def store(self, key, sample, words, residual_mean, residual_std):
        path = self._file(key)
        packed = pack_driving_sample(sample, words, residual_mean, residual_std, self.cfg)
        fd, temporary = tempfile.mkstemp(prefix=".tmp-", suffix=".pt", dir=str(self.path))
        os.close(fd)
        try:
            torch.save(packed, temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return path

    def close(self):
        """关闭切片缓存；切片文件本身采用原子替换写入。"""
        return None


class DrivingSliceDataset(Dataset):
    """直接读取 3 秒切片缓存，不执行二次数据预处理。"""

    def __init__(self, cfg):
        self.cache = DrivingSliceCache(cfg)
        self.files = sorted(self.cache.path.glob("*.pt"))
        if not self.files:
            raise FileNotFoundError("驾驶训练缓存为空: {}".format(self.cache.path))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = torch.load(self.files[index], map_location="cpu", weights_only=True)
        if not isinstance(sample, dict):
            raise ValueError("驾驶缓存条目必须是字典: {}".format(self.files[index]))
        return sample


def trajectory_to_planar_accelerations(trajectory, ego_state, dt_s: float):
    """由局部航点和局部初始二维速度计算可精确重建航点的加速度。"""
    trajectory = torch.as_tensor(trajectory).float()
    if trajectory.ndim < 2 or trajectory.shape[-1] != 2:
        raise ValueError("trajectory 必须为 [...,30,2]")
    if trajectory.shape[-2] != 30:
        raise ValueError("trajectory 必须包含 30 个航点")
    displacement = torch.cat((trajectory[..., :1, :], trajectory[..., 1:, :] - trajectory[..., :-1, :]), dim=-2)
    velocity = displacement / dt_s
    state = torch.as_tensor(ego_state).to(velocity)
    first = state[..., None, :2]
    return (velocity - torch.cat((first, velocity[..., :-1, :]), dim=-2)) / dt_s


def standardize_actions(actions, mean, std):
    """在动作通道上执行训练集均值方差标准化。"""
    mean = torch.as_tensor(mean, dtype=actions.dtype, device=actions.device)
    std = torch.as_tensor(std, dtype=actions.dtype, device=actions.device).clamp_min(1e-6)
    return (actions - mean) / std


def unstandardize_actions(actions, mean, std):
    """恢复物理二维加速度，仅供推理后处理使用。"""
    mean = torch.as_tensor(mean, dtype=actions.dtype, device=actions.device)
    std = torch.as_tensor(std, dtype=actions.dtype, device=actions.device)
    return actions * std + mean


def navigation_command_from_route(route_geometry, route_index, cfg):
    """读取前方路线的明确道路选项，缺失时从车道变化和路线转角推断。"""
    check_navigation_route(route_geometry, route_index)
    return _route_command(
        route_geometry, route_index, cfg.data.driving_cache.navigation_lookahead_m,
        cfg.data.driving_cache.navigation_turn_threshold_deg)


def _sample_expert(sample):
    """从当前帧或显式专家快照读取标注，不推断未提供的标签。"""
    return (sample.get("frame_meta") or {}).get("expert") or sample.get("expert") or {}


def _risk_targets(expert):
    """只缓存冲突概率与左右车道可用性，缺失值由逐维掩码排除。"""
    risk, lane = expert.get("risk") or {}, expert.get("lane_change") or {}
    conflict_value = risk.get("predicted_conflict_probability")
    conflict = float(conflict_value) if conflict_value is not None else float("nan")
    probability_valid = torch.tensor([
        math.isfinite(conflict) and 0.0 <= conflict <= 1.0,
        "available_left" in lane, "available_right" in lane,
    ], dtype=torch.bool)
    probabilities = torch.tensor([
        conflict if probability_valid[0] else 0.0,
        float(bool(lane.get("available_left", False))),
        float(bool(lane.get("available_right", False))),
    ])
    return probabilities, probability_valid


def _annotation_tensors(expert, spec):
    """校验采集器编号并缓存专家类别及三个带有效掩码的风险目标。"""
    state_id = int(expert.get("state_id", -1))
    reason_id = int(expert.get("reason_code", -1))
    if state_id != -1 and not 0 <= state_id < spec.annotation_state_classes:
        raise ValueError("专家驾驶状态编号超出当前标注类别: {}".format(state_id))
    if reason_id != -1 and not 0 <= reason_id < spec.annotation_reason_classes:
        raise ValueError("专家决策原因编号超出当前标注类别: {}".format(reason_id))
    probabilities, probability_valid = _risk_targets(expert)
    return {
        "annotation_valid": torch.tensor(float(bool(expert)), dtype=torch.float32),
        "annotation_state": torch.tensor(state_id, dtype=torch.long),
        "annotation_reason": torch.tensor(reason_id, dtype=torch.long),
        "annotation_probabilities": probabilities,
        "annotation_probability_valid": probability_valid.float(),
    }


def pack_driving_sample(sample, words, residual_mean, residual_std, cfg):
    """按当前物理速度选笛卡尔最近词，仅缓存该词的标准化物理加速度残差。"""
    if is_navigation_goal_approach({"expert": _sample_expert(sample)}):
        raise ValueError("Navigation goal approach is excluded from driving supervision")
    required = ("rgb", "intrinsics", "ego_state", "motion_matrices",
                "trajectory", "bev_grid", "timestamp", "scene", "index",
                "word_risks", "gt_risks")
    missing = [name for name in required if name not in sample]
    if missing:
        raise KeyError("驾驶缓存缺少字段: {}".format(", ".join(missing)))
    packed = {name: value.detach().cpu() if isinstance(value, torch.Tensor) else value
              for name, value in sample.items() if name in required}
    packed.update(_annotation_tensors(_sample_expert(sample), cfg.model.unified_driving))
    command = sample.get("navigation_command")
    if command is None:
        command = (sample.get("frame_meta") or {}).get("navigation_command")
    if command is None:
        raise KeyError("新驾驶缓存必须提供 navigation_command")
    if isinstance(command, str):
        command = cfg.model.unified_driving.navigation_commands.index(command)
    command = int(command)
    frame = sample.get("frame_meta") or {}
    recovery = sample.get("recovery_command", frame.get("recovery_command"))
    if recovery is None:
        recovery = (_sample_expert(sample).get("navigation") or {}).get("recovery_command", 0)
    # Read interim scenes that encoded recovery in the route command only
    # when the preserved original command is available.
    if command in (5, 6):
        recovery = command - 4
        command = sample.get("raw_navigation_command", frame.get("raw_navigation_command"))
        if command is None:
            raise ValueError("Legacy recovery requires raw_navigation_command")
        command = int(command)
    if not 0 <= command < len(cfg.model.unified_driving.navigation_commands):
        raise ValueError("navigation_command 超出配置的导航类别范围")
    packed["navigation_command"] = torch.tensor(command, dtype=torch.long)
    recovery = int(recovery)
    if not 0 <= recovery < len(cfg.model.unified_driving.recovery_commands):
        raise ValueError("recovery_command must be none/right/left (0/1/2)")
    packed["recovery_command"] = torch.tensor(recovery, dtype=torch.long)
    raw_actions = sample.get("trajectory_actions")
    if raw_actions is None:
        raw_actions = trajectory_to_planar_accelerations(
            packed["trajectory"], packed["ego_state"], cfg.model.unified_driving.waypoint_dt_s)
    raw_actions = torch.as_tensor(raw_actions).float().cpu()
    words = torch.as_tensor(words, dtype=torch.float32)
    spec = cfg.model.unified_driving
    nearest = nearest_planar_action_words(
        torch.as_tensor(packed.pop("trajectory")).float()[None],
        torch.as_tensor(packed["ego_state"]).float()[None, :2], words,
        spec.waypoint_dt_s, spec.nearest_weight_floor, spec.nearest_decay_s)[0]
    packed["trajectory_word_index"] = nearest.long()
    packed["trajectory_residual"] = standardize_actions(
        raw_actions - words[nearest], residual_mean, residual_std)
    packed["word_risks"] = torch.as_tensor(packed["word_risks"], dtype=torch.bool)
    packed["gt_risks"] = torch.as_tensor(packed["gt_risks"], dtype=torch.bool)
    packed["bev_grid"] = packed["bev_grid"].float()
    # 速度和加速度只在缓存阶段做 Signed-Log，训练及模型内不做逆变换。
    packed["ego_state"] = signed_log(packed["ego_state"].float())
    packed["motion_matrices"] = packed["motion_matrices"].float()
    return packed


def slice_driving_samples(samples, slice_duration_s: float = 3.0):
    """按场景时间轴每隔指定秒数选取一个训练切片起点。"""
    if slice_duration_s <= 0:
        raise ValueError("slice_duration_s 必须为正数")
    grouped = {}
    for sample in samples:
        grouped.setdefault(str(sample["scene"]), []).append(sample)
    selected = []
    for scene_samples in grouped.values():
        ordered = sorted(scene_samples, key=lambda item: float(item["timestamp"]))
        next_time = float(ordered[0]["timestamp"])
        for sample in ordered:
            timestamp = float(sample["timestamp"])
            if timestamp + 1e-4 >= next_time:
                selected.append(sample)
                next_time += slice_duration_s
    return selected


def prepare_driving_cache(samples, cfg):
    """? 3 ???????????????"""
    cache = DrivingSliceCache(cfg)
    sliced = slice_driving_samples(samples, cfg.data.driving_cache.slice_duration_s)
    if not sliced:
        return {"namespace": cache.namespace, "path": str(cache.path), "samples": 0}
    vocab_path = resolve_repo_path(
        str(Path(cfg.trajectory_vocabulary.output_dir) / cfg.trajectory_vocabulary.action_output))
    if not vocab_path.is_file():
        raise FileNotFoundError("生成驾驶缓存前必须先生成动作词表: {}".format(vocab_path))
    with np.load(vocab_path, allow_pickle=False) as artifact:
        words = artifact["actions"].copy()
        residual_mean = torch.from_numpy(artifact["residual_mean"].copy())
        residual_std = torch.from_numpy(artifact["residual_std"].copy())
    written, skipped, skipped_goal = 0, 0, 0
    active_scene, active_reader = None, None
    goal_cutoff = None
    try:
        for sample in sliced:
            sample = dict(sample)
            scene = Path(sample["scene"])
            if not scene.is_dir():
                scene = resolve_repo_path(str(Path(cfg.trajectory_vocabulary.scene_root) / scene))
            try:
                if scene != active_scene:
                    if active_reader is not None:
                        active_reader.close()
                    active_reader = SceneTrajectoryRisk(scene, cfg)
                    active_scene = scene
                    goal_cutoff = navigation_goal_cutoff(
                        active_reader.kinematics, active_reader.frames)
                if not successful_scene_meta(active_reader.reader.meta):
                    skipped += 1
                    continue
                frame = active_reader.frames[int(sample["index"])]
                if not window_before_navigation_goal(
                        frame["sim_time"], cfg.model.unified_driving.num_waypoints *
                        cfg.model.unified_driving.waypoint_dt_s, goal_cutoff):
                    skipped_goal += 1
                    continue
                # Both independent commands come from the saved frame.
                sample["frame_meta"] = frame
                if "navigation_command" in frame:
                    sample["navigation_command"] = active_reader.reader.navigation_command(
                        int(sample["index"]), cfg)
                sample["recovery_command"] = active_reader.reader.recovery_command(
                    int(sample["index"]))
                sample["trajectory"] = torch.from_numpy(active_reader.gt_trajectory(
                    int(sample["index"]), cfg.model.unified_driving.num_waypoints))
                yaw = math.radians(float(frame["ego"]["transform"][5]))
                cosine, sine = math.cos(yaw), math.sin(yaw)
                frame_ego = frame["ego"]
                motion = torch.tensor((frame_ego["velocity"][:2],
                                       frame_ego["acceleration"][:2]), dtype=torch.float32)
                local_motion = torch.stack((cosine * motion[:, 0] + sine * motion[:, 1],
                                            -sine * motion[:, 0] + cosine * motion[:, 1]), dim=-1)
                sample["ego_state"] = local_motion.flatten()
                sample["trajectory_actions"] = trajectory_to_planar_accelerations(
                    sample["trajectory"], sample["ego_state"], cfg.model.unified_driving.waypoint_dt_s)
                sample["word_risks"], sample["gt_risks"] = active_reader.evaluate(sample, words)
            except (FileNotFoundError, KeyError, OSError, ValueError, IndexError):
                skipped += 1
                continue
            key = "{}:{}".format(sample["scene"], sample["index"])
            cache.store(key, sample, words, residual_mean, residual_std)
            written += 1
    finally:
        if active_reader is not None:
            active_reader.close()
        cache.close()
    return {"namespace": cache.namespace, "path": str(cache.path), "samples": written,
            "skipped_missing_risk_gt": skipped,
            "skipped_navigation_goal": skipped_goal,
            "residual_mean": residual_mean.tolist(), "residual_std": residual_std.tolist()}
