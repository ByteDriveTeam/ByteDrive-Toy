"""把逐帧模型轨迹融合为世界系滚动计划，并转换为 CARLA 低层控制。

模块: clone_loop/control/control.py
依赖: math, numpy, clone_loop.control.checks.control_checks
读取配置:
    clone_loop.control.waypoint_dt_s / commit_horizon_s / blend_horizon_s /
        max_tracking_error_m / speed_horizon / min_target_speed_mps / max_target_speed_mps
    clone_loop.control.lookahead_min_m / lookahead_max_m / lookahead_time_s /
        lookahead_curvature_gain / wheelbase_m / max_steer_angle_deg / turn_steer_gain /
        steer_smoothing
    clone_loop.control.longitudinal_kp / longitudinal_ki / longitudinal_kd / integral_limit
    clone_loop.control.max_throttle / max_brake / brake_deadband_mps
    clone_loop.control.behavior_stop_threshold / behavior_stop_release_threshold /
        behavior_stop_indices
    clone_loop.simulation.fixed_delta_seconds（由构造参数传入）
对外接口:
    - TrajectoryController(cfg_control, fixed_delta_seconds)
        .reset() -> None
        .command(trajectory, pose, sim_time_s, speed_mps,
                 behavior_probabilities) -> tuple[dict, dict]
说明: Winner 轨迹先锚定到世界系；近端承诺旧计划、过渡段融合新预测、远端采用新预测。
      纯追踪前视距离由速度、曲率和计划稳定时域共同决定；障碍/红灯停车标签经双阈值状态机控制制动。
"""

import math

import numpy as np

from clone_loop.control.checks.control_checks import check_control_inputs


__all__ = ["TrajectoryController"]


class TrajectoryController:
    """模型轨迹到车辆执行器的有状态控制器。"""

    def __init__(self, cfg_control, fixed_delta_seconds):
        self._cfg = cfg_control
        self._dt = fixed_delta_seconds
        self.reset()

    def reset(self):
        """清空跨 episode 的滚动计划、PID 与转向平滑状态。"""
        self._active_times = None
        self._active_world = None
        self._last_sim_time = None
        self._stop_requested = False
        self._integral = 0.0
        self._previous_error = 0.0
        self._previous_steer = 0.0

    def command(self, trajectory, pose, sim_time_s, speed_mps,
                behavior_probabilities):
        """滚动融合 Winner 轨迹，并返回执行器命令与可诊断的实际参考轨迹。"""
        path = np.asarray(trajectory, dtype=np.float64)
        ego_pose = np.asarray(pose, dtype=np.float64)
        behaviors = np.asarray(behavior_probabilities, dtype=np.float64)
        check_control_inputs(
            path, ego_pose, sim_time_s, speed_mps,
            behaviors, self._cfg.behavior_stop_indices)
        now = float(sim_time_s)
        if self._last_sim_time is not None and now <= self._last_sim_time:
            raise ValueError("sim_time_s 必须在同一 episode 内严格递增")

        fresh_times, fresh_world = self._world_plan(path, ego_pose, now)
        active_world, diagnostics = self._roll_plan(fresh_times, fresh_world, ego_pose)
        reference = _world_to_ego(active_world[1:], ego_pose)
        active_target_speed = self._target_speed(active_world)
        fresh_target_speed = self._target_speed(fresh_world)
        stop_probability = self._update_stop_state(behaviors)
        target_speed = 0.0 if self._stop_requested else min(
            active_target_speed, fresh_target_speed)
        steer, lookahead, path_curvature = self._steer(reference, float(speed_mps))
        throttle, brake = self._longitudinal(target_speed, float(speed_mps))
        self._active_times = fresh_times
        self._active_world = active_world
        self._last_sim_time = now
        control = {
            "throttle": throttle, "steer": steer, "brake": brake,
            "target_speed_mps": target_speed,
        }
        execution = {
            "reference_trajectory": reference.astype(np.float32),
            "tracking_error_m": diagnostics["tracking_error_m"],
            "prediction_shift_m": diagnostics["prediction_shift_m"],
            "plan_reseeded": diagnostics["plan_reseeded"],
            "reseed_reason": diagnostics["reseed_reason"],
            "committed_waypoints": diagnostics["committed_waypoints"],
            "fresh_target_speed_mps": fresh_target_speed,
            "target_speed_mps": target_speed,
            "lookahead_m": lookahead,
            "path_curvature_inv_m": path_curvature,
            "stop_requested": self._stop_requested,
            "stop_probability": stop_probability,
        }
        return control, execution

    def _world_plan(self, path, pose, sim_time_s):
        """把当前 ego 原点与未来航点组成等间隔世界系计划。"""
        local = np.vstack((np.zeros((1, 2), dtype=np.float64), path))
        times = sim_time_s + self._cfg.waypoint_dt_s * np.arange(len(local))
        return times, _ego_to_world(local, pose)

    def _roll_plan(self, fresh_times, fresh_world, pose):
        """在统一时间轴上承诺、融合或按跟踪状态重建 active plan。"""
        if self._active_world is None:
            return fresh_world, self._diagnostics(0.0, 0.0, True, "initial", 0)

        now = float(fresh_times[0])
        old_now = _interpolate_plan(self._active_times, self._active_world, np.array([now]))[0]
        tracking_error = float(np.linalg.norm(old_now - pose[:2]))
        overlap = fresh_times <= self._active_times[-1]
        old_aligned = _interpolate_plan(self._active_times, self._active_world, fresh_times)
        prediction_shift = float(np.mean(np.linalg.norm(
            old_aligned[overlap] - fresh_world[overlap], axis=1)))
        reason = self._reseed_reason(now, fresh_times, tracking_error)
        if reason is not None:
            return fresh_world, self._diagnostics(
                tracking_error, prediction_shift, True, reason, 0)

        offsets = fresh_times - now
        commit = self._cfg.commit_horizon_s
        blend = self._cfg.blend_horizon_s
        old_weight = np.where(
            offsets <= commit, 1.0,
            np.where(offsets < commit + blend,
                     1.0 - (offsets - commit) / blend, 0.0))
        active = old_weight[:, None] * old_aligned + (1.0 - old_weight[:, None]) * fresh_world
        committed = int(np.count_nonzero((offsets[1:] <= commit) & overlap[1:]))
        return active, self._diagnostics(
            tracking_error, prediction_shift, False, None, committed)

    def _reseed_reason(self, now, fresh_times, tracking_error):
        """只在旧计划已失去时间或空间参考价值时放弃短期承诺。"""
        elapsed = now - self._last_sim_time
        tolerance = np.sqrt(np.finfo(np.float64).eps) * max(abs(now), 1.0)
        if not np.isclose(elapsed, self._dt, rtol=0.0, atol=tolerance):
            return "time_gap"
        required_end = now + self._cfg.commit_horizon_s + self._cfg.blend_horizon_s
        if self._active_times[-1] < required_end or fresh_times[0] < self._active_times[0]:
            return "coverage"
        if tracking_error > self._cfg.max_tracking_error_m:
            return "tracking_error"
        return None

    @staticmethod
    def _diagnostics(tracking_error, prediction_shift, reseeded, reason, committed):
        return {
            "tracking_error_m": float(tracking_error),
            "prediction_shift_m": float(prediction_shift),
            "plan_reseeded": bool(reseeded),
            "reseed_reason": reason,
            "committed_waypoints": int(committed),
        }

    def _target_speed(self, world_plan):
        """由带当前参考点的世界系计划前段估计期望速度。"""
        points = world_plan[:self._cfg.speed_horizon + 1]
        segment_speeds = np.linalg.norm(
            np.diff(points, axis=0), axis=1) / self._cfg.waypoint_dt_s
        raw = float(np.median(segment_speeds))
        return float(np.clip(
            raw, self._cfg.min_target_speed_mps, self._cfg.max_target_speed_mps))

    def _steer(self, path, speed_mps):
        """按速度、路径曲率与稳定计划时域自动选择纯追踪前视点。"""
        path_curvature = _path_curvature(path)
        speed_lookahead = np.clip(
            self._cfg.lookahead_min_m + speed_mps * self._cfg.lookahead_time_s,
            self._cfg.lookahead_min_m, self._cfg.lookahead_max_m)
        curved_lookahead = max(
            speed_lookahead / (
                1.0 + self._cfg.lookahead_curvature_gain * path_curvature),
            self._cfg.lookahead_min_m)
        stable_time = self._cfg.commit_horizon_s + self._cfg.blend_horizon_s / 2.0
        stable_index = int(np.clip(
            round(stable_time / self._cfg.waypoint_dt_s) - 1, 0, len(path) - 1))
        stable_distance = float(np.linalg.norm(path[stable_index]))
        lookahead = min(float(curved_lookahead), stable_distance)
        distances = np.linalg.norm(path, axis=1)
        target = path[int(np.argmin(np.abs(distances - lookahead)))]
        lookahead_sq = max(float(np.dot(target, target)), np.finfo(np.float64).eps)
        curvature = 2.0 * float(target[1]) / lookahead_sq
        wheel_angle = math.atan(self._cfg.wheelbase_m * curvature)
        raw = float(np.clip(
            self._cfg.turn_steer_gain
            * wheel_angle / math.radians(self._cfg.max_steer_angle_deg), -1.0, 1.0))
        smooth = (self._cfg.steer_smoothing * self._previous_steer
                  + (1.0 - self._cfg.steer_smoothing) * raw)
        self._previous_steer = smooth
        return smooth, lookahead, path_curvature

    def _update_stop_state(self, behaviors):
        """以停车语义概率的双阈值状态机抑制逐帧进入/释放抖动。"""
        stop_probability = float(np.max(behaviors[self._cfg.behavior_stop_indices]))
        threshold = (
            self._cfg.behavior_stop_release_threshold
            if self._stop_requested else self._cfg.behavior_stop_threshold)
        self._stop_requested = (
            stop_probability >= threshold if not self._stop_requested
            else stop_probability > threshold)
        return stop_probability

    def _longitudinal(self, target_speed, speed):
        """速度误差 PID；正输出映射油门，负输出映射制动。"""
        error = target_speed - speed
        self._integral = float(np.clip(
            self._integral + error * self._dt,
            -self._cfg.integral_limit, self._cfg.integral_limit))
        derivative = (error - self._previous_error) / self._dt
        self._previous_error = error
        effort = (self._cfg.longitudinal_kp * error
                  + self._cfg.longitudinal_ki * self._integral
                  + self._cfg.longitudinal_kd * derivative)
        if target_speed <= self._cfg.brake_deadband_mps and speed > self._cfg.brake_deadband_mps:
            effort = min(effort, -self._cfg.max_brake)
        return (
            float(np.clip(effort, 0.0, self._cfg.max_throttle)),
            float(np.clip(-effort, 0.0, self._cfg.max_brake)),
        )


def _ego_to_world(points, pose):
    """把 CARLA 左手 ego xy 批量旋转平移到世界平面。"""
    yaw = math.radians(float(pose[5]))
    cosine, sine = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    return points @ rotation.T + pose[:2]


def _world_to_ego(points, pose):
    """把世界平面点批量转换到当前 CARLA 左手 ego xy。"""
    yaw = math.radians(float(pose[5]))
    cosine, sine = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[cosine, sine], [-sine, cosine]], dtype=np.float64)
    return (points - pose[:2]) @ rotation.T


def _interpolate_plan(times, points, target_times):
    """在单调时间轴上对世界系 xy 分别线性插值。"""
    return np.column_stack([
        np.interp(target_times, times, points[:, axis]) for axis in range(2)
    ])


def _path_curvature(path):
    """由相邻轨迹段航向变化估计稳健的绝对曲率中位数。"""
    segments = np.diff(np.vstack((np.zeros((1, 2)), path)), axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    headings = np.arctan2(segments[:, 1], segments[:, 0])
    turns = np.abs(np.arctan2(
        np.sin(np.diff(headings)), np.cos(np.diff(headings))))
    valid = lengths[1:] > np.finfo(np.float64).eps
    return float(np.median(turns[valid] / lengths[1:][valid])) if np.any(valid) else 0.0
