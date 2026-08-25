"""验证世界系滚动轨迹承诺、融合、重建、控制边界与配置约束。

模块: clone_loop/control/tests/test_control.py
依赖: dataclasses, unittest, numpy, config, config.schema, clone_loop.control
读取配置: config/default.yaml（经 load_config 加载完整测试配置）
对外接口: —
说明: 测试只执行纯数值逻辑，不连接 CARLA、不创建项目目录外的临时文件。
"""

from dataclasses import replace
import unittest

import numpy as np

from clone_loop.control import TrajectoryController
from clone_loop.control.control import _path_curvature
from config import load_config
from config.schema import validate_config


class TrajectoryControllerTests(unittest.TestCase):
    """覆盖持久计划的关键时间与空间不变量。"""

    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config()

    def setUp(self):
        self.controller = TrajectoryController(self.cfg.clone_loop.control, 0.1)
        self.pose = np.zeros(6, dtype=np.float64)
        self.behavior = np.zeros(8, dtype=np.float64)

    @staticmethod
    def _straight(speed=4.0):
        distance = speed * 0.1 * np.arange(1, 21, dtype=np.float64)
        return np.column_stack((distance, np.zeros_like(distance)))

    def test_world_anchor_survives_translation_and_rotation(self):
        """旧计划须保持世界系位置，而不是随新 ego 原点一起移动。"""
        path = self._straight()
        self.controller.command(path, self.pose, 0.0, 4.0, self.behavior)
        rotated_pose = np.array([0.4, 0.0, 0.0, 0.0, 0.0, 90.0])
        _, execution = self.controller.command(
            path, rotated_pose, 0.1, 4.0, self.behavior)
        np.testing.assert_allclose(
            execution["reference_trajectory"][0], [0.0, -0.4], atol=1e-7)
        self.assertEqual(execution["committed_waypoints"], 5)

    def test_commit_blend_and_fresh_sections(self):
        """近端五点保持旧轨迹，随后五点线性融合，远端采用新预测。"""
        old = self._straight()
        self.controller.command(old, self.pose, 0.0, 4.0, self.behavior)
        pose = np.array([0.4, 0.0, 0.0, 0.0, 0.0, 0.0])
        fresh = self._straight()
        fresh[:, 1] = 4.0
        _, execution = self.controller.command(
            fresh, pose, 0.1, 4.0, self.behavior)
        reference = execution["reference_trajectory"]
        np.testing.assert_allclose(reference[:5, 1], 0.0)
        self.assertAlmostEqual(float(reference[5, 1]), 0.8, places=6)
        self.assertAlmostEqual(float(reference[7, 1]), 2.4, places=6)
        np.testing.assert_allclose(reference[9:, 1], 4.0)

    def test_long_range_prediction_rolls_toward_control(self):
        """远端新预测写入 active plan 后，应在后续滚动中成为受承诺参考。"""
        path = self._straight()
        self.controller.command(path, self.pose, 0.0, 4.0, self.behavior)
        pose = np.array([0.4, 0.0, 0.0, 0.0, 0.0, 0.0])
        curved = self._straight()
        curved[:, 1] = np.linspace(0.0, 2.0, len(curved))
        self.controller.command(curved, pose, 0.1, 4.0, self.behavior)
        for step in range(2, 7):
            pose = np.array([0.4 * step, 0.0, 0.0, 0.0, 0.0, 0.0])
            _, execution = self.controller.command(
                curved, pose, 0.1 * step, 4.0, self.behavior)
        self.assertGreater(float(execution["reference_trajectory"][0, 1]), 0.0)

    def test_stop_label_overrides_committed_speed(self):
        """横向仍承诺旧计划时，模型停车标签也必须立即把目标速度降到零。"""
        fast = self._straight(speed=8.0)
        self.controller.command(fast, self.pose, 0.0, 0.0, self.behavior)
        pose = np.array([0.8, 0.0, 0.0, 0.0, 0.0, 0.0])
        stop_behavior = self.behavior.copy()
        stop_behavior[0] = 0.7
        command, execution = self.controller.command(
            fast, pose, 0.1, 8.0, stop_behavior)
        self.assertEqual(command["target_speed_mps"], 0.0)
        self.assertTrue(execution["stop_requested"])
        self.assertAlmostEqual(execution["stop_probability"], 0.7)
        self.assertGreater(command["brake"], 0.0)

    def test_stop_label_uses_release_hysteresis(self):
        """停车概率位于双阈值之间时保持停车，低于释放阈值后恢复轨迹速度。"""
        path = self._straight()
        enter = self.behavior.copy()
        enter[1] = 0.7
        command, _ = self.controller.command(path, self.pose, 0.0, 4.0, enter)
        self.assertEqual(command["target_speed_mps"], 0.0)
        hold = self.behavior.copy()
        hold[1] = 0.5
        pose = np.array([0.4, 0.0, 0.0, 0.0, 0.0, 0.0])
        command, execution = self.controller.command(path, pose, 0.1, 4.0, hold)
        self.assertTrue(execution["stop_requested"])
        release = self.behavior.copy()
        release[1] = 0.4
        pose = np.array([0.8, 0.0, 0.0, 0.0, 0.0, 0.0])
        command, execution = self.controller.command(path, pose, 0.2, 4.0, release)
        self.assertFalse(execution["stop_requested"])
        self.assertGreater(command["target_speed_mps"], 0.0)

    def test_faster_prediction_does_not_replace_committed_geometry(self):
        """新预测加速时，近端空间参考仍由旧计划决定。"""
        slow = self._straight(speed=2.0)
        self.controller.command(slow, self.pose, 0.0, 2.0, self.behavior)
        pose = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
        fast = self._straight(speed=8.0)
        _, execution = self.controller.command(
            fast, pose, 0.1, 2.0, self.behavior)
        expected = 0.2 * np.arange(1, 6, dtype=np.float32)
        np.testing.assert_allclose(execution["reference_trajectory"][:5, 0], expected)

    def test_tracking_error_and_time_gap_reseed(self):
        """大跟踪误差与丢 tick 都应放弃过时计划。"""
        path = self._straight()
        self.controller.command(path, self.pose, 0.0, 4.0, self.behavior)
        far_pose = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        _, tracking = self.controller.command(
            path, far_pose, 0.1, 4.0, self.behavior)
        self.assertTrue(tracking["plan_reseeded"])
        self.assertEqual(tracking["reseed_reason"], "tracking_error")
        _, gap = self.controller.command(
            path, far_pose, 0.3, 4.0, self.behavior)
        self.assertTrue(gap["plan_reseeded"])
        self.assertEqual(gap["reseed_reason"], "time_gap")

    def test_reset_and_non_monotonic_time(self):
        """reset 后首帧重新初始化，同一 episode 的时间倒退则硬失败。"""
        path = self._straight()
        self.controller.command(path, self.pose, 1.0, 0.0, self.behavior)
        with self.assertRaisesRegex(ValueError, "严格递增"):
            self.controller.command(path, self.pose, 1.0, 0.0, self.behavior)
        self.controller.reset()
        _, execution = self.controller.command(
            path, self.pose, 0.0, 0.0, self.behavior)
        self.assertEqual(execution["reseed_reason"], "initial")

    def test_control_bounds_and_input_validation(self):
        """执行器范围稳定，非法轨迹与位姿在控制入口拒绝。"""
        command, _ = self.controller.command(
            self._straight(), self.pose, 0.0, 0.0, self.behavior)
        self.assertTrue(0.0 <= command["throttle"] <= 1.0)
        self.assertTrue(-1.0 <= command["steer"] <= 1.0)
        self.assertTrue(0.0 <= command["brake"] <= 1.0)
        self.controller.reset()
        with self.assertRaises(ValueError):
            self.controller.command(
                np.array([[np.nan, 0.0]]), self.pose, 0.0, 0.0, self.behavior)
        with self.assertRaises(ValueError):
            self.controller.command(
                self._straight(), np.zeros(5), 0.0, 0.0, self.behavior)
        with self.assertRaises(ValueError):
            self.controller.command(
                self._straight(), self.pose, 0.0, -1.0, self.behavior)
        with self.assertRaises(ValueError):
            self.controller.command(
                self._straight(), self.pose, 0.0, 0.0, np.zeros(1))

    def test_lookahead_adapts_to_speed_curvature_and_stable_horizon(self):
        """前视距离应随速度增加、随曲率缩短，并不越过滚动计划稳定时域。"""
        straight = self._straight(speed=8.0)
        _, fast = self.controller.command(
            straight, self.pose, 0.0, 8.0, self.behavior)
        fast_lookahead = fast["lookahead_m"]
        stable_index = int(round((
            self.cfg.clone_loop.control.commit_horizon_s
            + self.cfg.clone_loop.control.blend_horizon_s / 2.0) / 0.1) - 1)
        self.assertLessEqual(
            fast_lookahead,
            float(np.linalg.norm(fast["reference_trajectory"][stable_index])))

        curved_controller = TrajectoryController(self.cfg.clone_loop.control, 0.1)
        curved = straight.copy()
        curved[:, 1] = 0.08 * curved[:, 0] ** 2
        _, turn = curved_controller.command(
            curved, self.pose, 0.0, 8.0, self.behavior)
        self.assertGreater(turn["path_curvature_inv_m"], 0.0)
        self.assertLess(turn["lookahead_m"], fast_lookahead)

    def test_rolling_reference_reduces_moving_goal_jitter(self):
        """同一自动前视规则下，滚动 reference 应显著抑制逐帧远端预测摆动。"""
        cfg = self.cfg.clone_loop.control
        x = np.arange(1, 21, dtype=np.float64) * 0.4
        stable_index = int(round((
            cfg.commit_horizon_s + cfg.blend_horizon_s / 2.0)
            / cfg.waypoint_dt_s) - 1)
        direct_targets, rolling_targets = [], []
        for step in range(30):
            noise = (1.0 if step % 2 == 0 else -1.0) * 0.9 * (x / x[-1])
            path = np.column_stack((x, 0.6 * (x / x[-1]) ** 2 + noise))
            curvature = _path_curvature(path)
            speed_lookahead = np.clip(
                cfg.lookahead_min_m + 4.0 * cfg.lookahead_time_s,
                cfg.lookahead_min_m, cfg.lookahead_max_m)
            direct_lookahead = max(
                speed_lookahead / (1.0 + cfg.lookahead_curvature_gain * curvature),
                cfg.lookahead_min_m)
            direct_lookahead = min(
                direct_lookahead, float(np.linalg.norm(path[stable_index])))
            direct_index = int(np.argmin(np.abs(
                np.linalg.norm(path, axis=1) - direct_lookahead)))
            direct_targets.append(path[direct_index, 1])

            pose = np.array([step * 0.4, 0.0, 0.0, 0.0, 0.0, 0.0])
            _, execution = self.controller.command(
                path, pose, step * 0.1, 4.0, self.behavior)
            reference = execution["reference_trajectory"]
            rolling_index = int(np.argmin(np.abs(
                np.linalg.norm(reference, axis=1) - execution["lookahead_m"])))
            rolling_targets.append(reference[rolling_index, 1])

        direct_jitter = float(np.std(np.diff(direct_targets)))
        rolling_jitter = float(np.std(np.diff(rolling_targets)))
        self.assertLess(rolling_jitter, direct_jitter * 0.4)


class TrajectoryControlConfigTests(unittest.TestCase):
    """覆盖滚动执行新增配置的加载期约束。"""

    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config()

    def _validate_control(self, **changes):
        control = replace(self.cfg.clone_loop.control, **changes)
        clone_loop = replace(self.cfg.clone_loop, control=control)
        validate_config(replace(self.cfg, clone_loop=clone_loop))

    def test_horizon_must_align_and_leave_fresh_tail(self):
        """承诺窗口须对齐 10Hz，且不能吃掉完整预测时域。"""
        with self.assertRaises(AssertionError):
            self._validate_control(commit_horizon_s=0.55)
        with self.assertRaises(AssertionError):
            self._validate_control(commit_horizon_s=1.0, blend_horizon_s=1.0)
        with self.assertRaises(AssertionError):
            self._validate_control(max_tracking_error_m=float("inf"))

    def test_speed_horizon_cannot_exceed_trajectory(self):
        """速度估计窗口不得超过模型实际输出的航点数。"""
        with self.assertRaises(AssertionError):
            self._validate_control(speed_horizon=21)

    def test_adaptive_lookahead_and_stop_thresholds_are_bounded(self):
        """自动前视参数须有限，停车状态机释放阈值须严格低于进入阈值。"""
        with self.assertRaises(AssertionError):
            self._validate_control(lookahead_time_s=float("inf"))
        with self.assertRaises(AssertionError):
            self._validate_control(
                behavior_stop_threshold=0.6,
                behavior_stop_release_threshold=0.6)


if __name__ == "__main__":
    unittest.main()
