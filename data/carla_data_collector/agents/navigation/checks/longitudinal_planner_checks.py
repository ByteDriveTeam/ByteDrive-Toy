"""Behavior checks for the short-horizon longitudinal planner."""

import math
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
from agents.navigation.longitudinal_planner import plan_longitudinal_speed


class LongitudinalPlannerChecks(unittest.TestCase):
    def test_stop_line_is_previewed_before_current_speed_cap(self):
        plan = plan_longitudinal_speed(8.0, 0.0, 12.0,
                                       [(22.0, 0.0)], step_s=0.1)
        free = plan_longitudinal_speed(8.0, 0.0, 12.0, [], step_s=0.1)
        self.assertLess(plan["target_speed_mps"], 8.0)
        self.assertGreater(free["target_speed_mps"], 8.0)
        self.assertEqual(len(plan["samples"]), 30)
        self.assertTrue(all(sample["speed_mps"] >= 0.0
                            for sample in plan["samples"]))

    def test_braking_starts_earlier_for_a_nearer_stop(self):
        far = plan_longitudinal_speed(8.0, 0.0, 12.0,
                                      [(38.0, 0.0)], step_s=0.1)
        near = plan_longitudinal_speed(8.0, 0.0, 12.0,
                                       [(22.0, 0.0)], step_s=0.1)
        self.assertLess(near["target_speed_mps"], far["target_speed_mps"])
        self.assertLess(near["samples"][-1]["route_distance_m"], 22.0)

    def test_stopped_ego_can_close_gap_to_stop_line(self):
        plan = plan_longitudinal_speed(
            0.0, 0.0, 3.4, [(2.0, 0.0)], step_s=0.1)
        self.assertGreater(plan["target_speed_mps"], 0.2)
        self.assertLess(plan["samples"][-1]["route_distance_m"], 2.0)
        self.assertLess(plan["samples"][-1]["speed_mps"], 0.2)

    def test_low_speed_creep_reaches_cap_without_freezing(self):
        plan = plan_longitudinal_speed(0.0, 0.0, 1.0, [], step_s=0.1)
        self.assertGreater(plan["target_speed_mps"], 0.2)
        self.assertGreater(plan["samples"][-1]["speed_mps"], 0.7)
        self.assertLess(plan["samples"][-1]["speed_mps"], 1.3)

    def test_acceleration_and_jerk_limits(self):
        plan = plan_longitudinal_speed(5.0, 0.0, 12.0,
                                       [(18.0, 3.0)], step_s=0.1)
        old_accel = 0.0
        for sample in plan["samples"]:
            accel = sample["acceleration_mps2"]
            self.assertLessEqual(accel, 2.2 + 1e-9)
            self.assertGreaterEqual(accel, -3.0 - 1e-9)
            self.assertLessEqual(abs(accel - old_accel), 4.0 * 0.1 + 1e-9)
            old_accel = accel

    def test_fixed_input_produces_identical_plan(self):
        args = (6.0, -0.4, 9.0, [(16.0, 4.0), (25.0, 0.0)])
        self.assertEqual(plan_longitudinal_speed(*args),
                         plan_longitudinal_speed(*args))

    def test_receding_horizon_stops_before_stop_line(self):
        speed = 8.0
        acceleration = 0.0
        travelled = 0.0
        for _ in range(100):
            distance = max(22.0 - travelled, 0.0)
            current_cap = math.sqrt(2.0 * 3.0 * max(
                distance - 2.0 - 0.7 * speed, 0.0))
            plan = plan_longitudinal_speed(
                speed, acceleration, current_cap,
                [(distance, 0.0)], step_s=0.1)
            acceleration = plan["target_acceleration_mps2"]
            next_speed = max(speed + acceleration * 0.1, 0.0)
            travelled += (speed + next_speed) * 0.05
            speed = next_speed
            if speed < 0.1:
                break
        self.assertLessEqual(travelled, 22.0)
        self.assertLess(speed, 0.1)


if __name__ == "__main__":
    unittest.main()
