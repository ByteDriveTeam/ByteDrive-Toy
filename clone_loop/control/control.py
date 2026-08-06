"""把模型选中的 ego 系轨迹转换为 CARLA 归一化转向、油门与制动。

模块: clone_loop/control/control.py
依赖: math, numpy, clone_loop.control.checks.control_checks
读取配置:
    clone_loop.control.waypoint_dt_s / speed_horizon / min_target_speed_mps / max_target_speed_mps
    clone_loop.control.lookahead_m / wheelbase_m / max_steer_angle_deg / turn_steer_gain /
        steer_smoothing
    clone_loop.control.longitudinal_kp / longitudinal_ki / longitudinal_kd / integral_limit
    clone_loop.control.max_throttle / max_brake / brake_deadband_mps
    clone_loop.simulation.fixed_delta_seconds（由构造参数传入）
对外接口:
    - TrajectoryController(cfg_control, fixed_delta_seconds)
        .reset() -> None
        .command(trajectory, speed_mps) -> dict
说明: 横向采用纯追踪几何，纵向采用带积分限幅的 PID；目标速度只从 Winner 轨迹推导，
      行为标签不得参与控制，模型停车意图必须由近零位移轨迹表达。
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
        """清空跨 episode 的 PID 与转向平滑状态。"""
        self._integral = 0.0
        self._previous_error = 0.0
        self._previous_steer = 0.0

    def command(self, trajectory, speed_mps):
        """只根据 Winner 轨迹与当前速度生成 CARLA VehicleControl 标量字典。"""
        path = np.asarray(trajectory, dtype=np.float64)
        check_control_inputs(path, speed_mps)
        target_speed = self._target_speed(path)
        steer = self._steer(path)
        throttle, brake = self._longitudinal(target_speed, float(speed_mps))
        return {
            "throttle": throttle, "steer": steer, "brake": brake,
            "target_speed_mps": target_speed,
        }

    def _target_speed(self, path):
        """由训练帧间隔下的前若干航点位移估计期望速度。"""
        points = np.vstack((np.zeros((1, 2)), path[:self._cfg.speed_horizon]))
        segment_speeds = np.linalg.norm(np.diff(points, axis=0), axis=1) / self._cfg.waypoint_dt_s
        raw = float(np.median(segment_speeds))
        return float(np.clip(
            raw, self._cfg.min_target_speed_mps, self._cfg.max_target_speed_mps))

    def _steer(self, path):
        """选择最接近期望前视距离的航点，并用纯追踪曲率求转向。"""
        distances = np.linalg.norm(path, axis=1)
        target = path[int(np.argmin(np.abs(distances - self._cfg.lookahead_m)))]
        lookahead_sq = max(float(np.dot(target, target)), np.finfo(np.float64).eps)
        curvature = 2.0 * float(target[1]) / lookahead_sq
        wheel_angle = math.atan(self._cfg.wheelbase_m * curvature)
        raw = float(np.clip(
            self._cfg.turn_steer_gain
            * wheel_angle / math.radians(self._cfg.max_steer_angle_deg), -1.0, 1.0))
        smooth = (self._cfg.steer_smoothing * self._previous_steer
                  + (1.0 - self._cfg.steer_smoothing) * raw)
        self._previous_steer = smooth
        return smooth

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
