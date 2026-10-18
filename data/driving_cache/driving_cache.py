"""新驾驶训练切片缓存：把训练输入打包为可直接读取的 Tensor 字典。

模块: data/driving_cache/driving_cache.py
依赖: hashlib, json, pathlib, tempfile, torch
读取配置: data.driving_cache.dir/version/slice_duration_s/navigation_lookahead_m/
          navigation_turn_threshold_deg/navigation_lane_match_radius_m/
          navigation_heading_tolerance_deg/navigation_lane_stable_points,
          data.scene_cache_size / driving.scene_root,
          model.unified_driving.navigation_commands
对外接口:
    - DrivingSliceCache(cfg) -> object
    - load(key) -> dict[str, Tensor] | None
    - store(key, sample) -> Path
    - close() -> None
    - navigation_command_from_route(route_geometry, route_index, cfg) -> int
    - pack_driving_sample(sample, action_mean, action_std, cfg) -> dict[str, Tensor]
说明: 路线选项优先决定五类指令；旧 LMDB 经 HDMap 恢复变道，缓存只保存类别 ID；轨迹控制量在训练集
      Z-Score 空间写入，BEV latent 在线计算。
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path
import tempfile

import torch
from torch.utils.data import Dataset

from data.carla_data_collector.common.navigation_command import (
    navigation_command_from_route as _route_command,
)
from data.driving_cache.checks.driving_cache_checks import check_navigation_route
from data.single_frame_base import resolve_repo_path
from model.driving_model.unified_driving_model import signed_log


__all__ = ["DrivingSliceCache", "DrivingSliceDataset", "trajectory_to_unicycle_actions",
           "standardize_actions", "unstandardize_actions", "pack_driving_sample",
           "navigation_command_from_route", "slice_driving_samples", "prepare_driving_cache"]


_ACTION_DT_S = 0.1
_ACTION_EPS = 1e-3


class DrivingSliceCache:
    """版本隔离、原子写入的 3 秒驾驶样本缓存。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self._navigation_readers = OrderedDict()
        self.root = resolve_repo_path(cfg.data.driving_cache.dir)
        self.namespace = self._namespace(cfg.data.driving_cache.version, cfg)
        self.path = self.root / self.namespace
        self.path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _namespace(version, cfg):
        payload = {"version": version, "slice_duration_s": float(cfg.data.driving_cache.slice_duration_s),
                   "trajectory_target": "standardized_unicycle_accel_curvature_v1",
                   "navigation_commands": cfg.model.unified_driving.navigation_commands,
                   "navigation_lookahead_m": cfg.data.driving_cache.navigation_lookahead_m,
                   "navigation_turn_threshold_deg": cfg.data.driving_cache.navigation_turn_threshold_deg,
                   "navigation_lane_match_radius_m": cfg.data.driving_cache.navigation_lane_match_radius_m,
                   "navigation_heading_tolerance_deg": cfg.data.driving_cache.navigation_heading_tolerance_deg,
                   "navigation_lane_stable_points": cfg.data.driving_cache.navigation_lane_stable_points,
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

    def store(self, key, sample, action_mean=None, action_std=None):
        path = self._file(key)
        if sample.get("navigation_command") is None \
                and (sample.get("frame_meta") or {}).get("navigation_command") is None \
                and "route_geometry" not in sample:
            scene_dir = _scene_dir(sample, self.cfg)
            if (scene_dir / "lmdb").is_dir():
                reader = self._navigation_readers.pop(scene_dir, None)
                if reader is None:
                    from vis.data_vis.reader import SceneReader
                    reader = SceneReader(scene_dir, open_videos=False)
                self._navigation_readers[scene_dir] = reader
                while len(self._navigation_readers) > self.cfg.data.scene_cache_size:
                    self._navigation_readers.popitem(last=False)[1].close()
                sample = dict(sample, scene_reader=reader)
        packed = pack_driving_sample(sample, action_mean, action_std, self.cfg)
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
        """释放缓存预生成阶段复用的场景读取器。"""
        for reader in self._navigation_readers.values():
            reader.close()
        self._navigation_readers.clear()


class DrivingSliceDataset(Dataset):
    """直接读取 3 秒切片缓存，不执行二次数据预处理。"""

    def __init__(self, cfg):
        self.cache = DrivingSliceCache(cfg)
        stats_path = self.cache.path / "action_stats.pt"
        self.action_stats = torch.load(stats_path, map_location="cpu", weights_only=True) \
            if stats_path.is_file() else None
        self.files = sorted(self.cache.path.glob("*.pt"))
        self.files = [path for path in self.files if path.name != "action_stats.pt"]
        if not self.files:
            raise FileNotFoundError("驾驶训练缓存为空: {}".format(self.cache.path))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = torch.load(self.files[index], map_location="cpu", weights_only=True)
        if not isinstance(sample, dict):
            raise ValueError("驾驶缓存条目必须是字典: {}".format(self.files[index]))
        return sample


def trajectory_to_unicycle_actions(trajectory, ego_state=None, dt_s: float = _ACTION_DT_S):
    """从局部航点计算 [加速度, 曲率] 控制序列。"""
    trajectory = torch.as_tensor(trajectory).float()
    if trajectory.ndim < 2 or trajectory.shape[-1] != 2:
        raise ValueError("trajectory 必须为 [...,30,2]")
    if trajectory.shape[-2] != 30:
        raise ValueError("trajectory 必须包含 30 个航点")
    displacement = torch.cat((trajectory[..., :1, :], trajectory[..., 1:, :] - trajectory[..., :-1, :]), dim=-2)
    velocity = displacement / dt_s
    speed = velocity.norm(dim=-1)
    if ego_state is None:
        initial_speed = speed[..., :1].detach().zero_()
    else:
        state = torch.as_tensor(ego_state).float()
        initial_speed = state[..., :2].norm(dim=-1, keepdim=True)
    acceleration = (speed - torch.cat((initial_speed, speed[..., :-1]), dim=-1)) / dt_s
    heading = torch.atan2(velocity[..., 1], velocity[..., 0])
    heading_delta = torch.atan2(
        torch.sin(heading - torch.cat((heading[..., :1].detach().zero_(), heading[..., :-1]), dim=-1)),
        torch.cos(heading - torch.cat((heading[..., :1].detach().zero_(), heading[..., :-1]), dim=-1)))
    curvature = heading_delta / (speed.clamp_min(_ACTION_EPS) * dt_s)
    return torch.stack((acceleration, curvature), dim=-1)


def standardize_actions(actions, mean, std):
    """在动作通道上执行训练集均值方差标准化。"""
    mean = torch.as_tensor(mean, dtype=actions.dtype, device=actions.device)
    std = torch.as_tensor(std, dtype=actions.dtype, device=actions.device).clamp_min(1e-6)
    return (actions - mean) / std


def unstandardize_actions(actions, mean, std):
    """恢复物理加速度和曲率，仅供推理后处理使用。"""
    mean = torch.as_tensor(mean, dtype=actions.dtype, device=actions.device)
    std = torch.as_tensor(std, dtype=actions.dtype, device=actions.device)
    return actions * std + mean


def navigation_command_from_route(route_geometry, route_index, cfg):
    """读取前方路线的明确道路选项，缺失时从车道变化和路线转角推断。"""
    check_navigation_route(route_geometry, route_index)
    return _route_command(
        route_geometry, route_index, cfg.data.driving_cache.navigation_lookahead_m,
        cfg.data.driving_cache.navigation_turn_threshold_deg)


def _navigation_command_from_trajectory(trajectory, cfg):
    """脱离场景文件的旧样本仅能从局部航点近似恢复直行或转弯。"""
    waypoints = torch.as_tensor(trajectory, dtype=torch.float32)[:, :2]
    points = torch.cat((torch.zeros_like(waypoints[:1]), waypoints), dim=0)
    distances = (points[1:] - points[:-1]).norm(dim=-1)
    arc = torch.cat((distances.new_zeros(1), distances.cumsum(0)))
    return _route_command(
        {"points": points.tolist(), "arc_m": arc.tolist()}, 0,
        cfg.data.driving_cache.navigation_lookahead_m,
        cfg.data.driving_cache.navigation_turn_threshold_deg)


def _scene_dir(sample, cfg):
    """解析旧样本的场景目录，支持绝对路径和 scene_* 名称。"""
    scene_dir = Path(str(sample["scene"]))
    if not scene_dir.is_dir():
        scene_dir = resolve_repo_path(cfg.data.driving.scene_root) / scene_dir
    return scene_dir.resolve()


def _navigation_command_from_scene(sample, cfg):
    """旧 LMDB 场景优先使用 HDMap 和完整未来运动学恢复五类导航指令。"""
    reader = sample.get("scene_reader")
    if reader is not None:
        return reader.navigation_command(int(sample["index"]), cfg)
    scene_dir = _scene_dir(sample, cfg)
    if not (scene_dir / "lmdb").is_dir():
        return None
    from vis.data_vis.reader import SceneReader

    reader = SceneReader(scene_dir, open_videos=False)
    try:
        return reader.navigation_command(int(sample["index"]), cfg)
    finally:
        reader.close()


def pack_driving_sample(sample, action_mean=None, action_std=None, cfg=None):
    """打包训练必需字段，并把路线转成指令、航点转成标准化控制量。"""
    required = ("rgb", "intrinsics", "ego_state", "motion_matrices",
                "trajectory", "bev_grid", "timestamp", "scene", "index")
    missing = [name for name in required if name not in sample]
    if missing:
        raise KeyError("驾驶缓存缺少字段: {}".format(", ".join(missing)))
    packed = {name: value.detach().cpu() if isinstance(value, torch.Tensor) else value
              for name, value in sample.items() if name in required}
    command = sample.get("navigation_command")
    if command is None:
        command = (sample.get("frame_meta") or {}).get("navigation_command")
    if command is None:
        if cfg is None:
            raise KeyError("缺少 navigation_command，且未提供旧数据回退所需的 cfg")
        if "route_geometry" in sample and "route_index" in sample:
            command = navigation_command_from_route(
                sample["route_geometry"], sample["route_index"], cfg)
        else:
            command = _navigation_command_from_scene(sample, cfg)
            if command is None:
                command = _navigation_command_from_trajectory(packed["trajectory"], cfg)
    if isinstance(command, str):
        if cfg is None:
            raise ValueError("字符串导航指令需要 cfg.model.unified_driving.navigation_commands")
        command = cfg.model.unified_driving.navigation_commands.index(command)
    command = int(command)
    if not 0 <= command < 5:
        raise ValueError("navigation_command 必须是 0..4 的类别 ID")
    packed["navigation_command"] = torch.tensor(command, dtype=torch.long)
    raw_actions = sample.get("trajectory_actions")
    if raw_actions is None:
        raw_actions = trajectory_to_unicycle_actions(
            packed["trajectory"], packed["ego_state"], _ACTION_DT_S)
    raw_actions = torch.as_tensor(raw_actions).float()
    if action_mean is None:
        action_mean = (0.0, 0.0)
    if action_std is None:
        action_std = (1.0, 1.0)
    packed["trajectory"] = standardize_actions(raw_actions, action_mean, action_std)
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


def _fit_action_stats(samples):
    actions = [trajectory_to_unicycle_actions(sample["trajectory"], sample.get("ego_state"))
               for sample in samples]
    values = torch.cat([action.reshape(-1, 2) for action in actions], dim=0)
    mean = values.mean(dim=0)
    std = values.std(dim=0, unbiased=False).clamp_min(1e-3)
    return mean, std


def prepare_driving_cache(samples, cfg):
    """? 3 ???????????????"""
    cache = DrivingSliceCache(cfg)
    sliced = slice_driving_samples(samples, cfg.data.driving_cache.slice_duration_s)
    if not sliced:
        return {"namespace": cache.namespace, "path": str(cache.path), "samples": 0}
    action_mean, action_std = _fit_action_stats(sliced)
    stats_path = cache.path / "action_stats.pt"
    torch.save({"mean": action_mean, "std": action_std}, stats_path)
    written = 0
    try:
        for sample in sliced:
            key = "{}:{}".format(sample["scene"], sample["index"])
            cache.store(key, sample, action_mean, action_std)
            written += 1
    finally:
        cache.close()
    return {"namespace": cache.namespace, "path": str(cache.path), "samples": written,
            "action_mean": action_mean.tolist(), "action_std": action_std.tolist()}
