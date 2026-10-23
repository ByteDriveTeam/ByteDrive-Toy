"""Regression checks for BehaviorAgent obstacle classification."""

import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.local_planner import RoadOption


class PedestrianAvoidanceChecks(unittest.TestCase):
    def test_no_nearby_walkers_does_not_query_all_vehicles(self):
        agent = BehaviorAgent.__new__(BehaviorAgent)
        agent._world = SimpleNamespace(get_actors=lambda: SimpleNamespace(
            filter=lambda pattern: []))
        agent._vehicle_obstacle_detected = Mock(
            side_effect=AssertionError("empty walker list fell back to vehicles"))

        result = agent.pedestrian_avoid_manager(None)

        self.assertEqual(result, (False, None, -1))
        agent._vehicle_obstacle_detected.assert_not_called()

    def test_rear_traffic_does_not_rewrite_navigation_route(self):
        agent = BehaviorAgent.__new__(BehaviorAgent)
        agent._world = SimpleNamespace(get_actors=lambda: SimpleNamespace(
            filter=lambda pattern: []))
        agent._vehicle = SimpleNamespace(id=1)
        agent._behavior = SimpleNamespace(min_proximity_threshold=10)
        agent._speed = 20
        agent._speed_limit = 50
        agent._direction = RoadOption.LANEFOLLOW
        agent._vehicle_obstacle_detected = Mock(return_value=(False, None, -1))
        agent.set_destination = Mock(side_effect=AssertionError(
            "rear traffic changed the navigation route"))

        result = agent.collision_and_car_avoid_manager(SimpleNamespace(
            is_junction=False, transform=SimpleNamespace(location=None)))

        self.assertEqual(result, (False, None, -1))
        agent.set_destination.assert_not_called()


class LongitudinalIntegrationChecks(unittest.TestCase):
    @staticmethod
    def _agent(speed_mps=0.0, target_mps=0.0, accel_mps2=0.0):
        agent = BehaviorAgent.__new__(BehaviorAgent)
        agent._speed = speed_mps * 3.6
        agent._last_target_speed_mps = target_mps
        agent._last_target_accel_mps2 = accel_mps2
        agent._longitudinal_horizon_s = 3.0
        agent._longitudinal_step_s = 0.25
        agent._longitudinal_preview_s = 0.5
        agent._max_accel_mps2 = 2.2
        agent._comfort_decel_mps2 = 3.0
        agent._max_jerk_mps3 = 4.0
        return agent

    def test_issued_target_respects_tick_acceleration_and_jerk(self):
        agent = self._agent()

        target_kmh = agent._plan_target_speed(36.0, [], 0.1)

        self.assertLessEqual(target_kmh / 3.6, 0.04 + 1e-9)
        self.assertLessEqual(agent._last_target_accel_mps2, 0.4 + 1e-9)
        self.assertGreater(agent._last_longitudinal_plan[
            "preview_speed_mps"], agent._last_target_speed_mps)

    def test_cleared_constraint_does_not_continue_braking_in_cruise(self):
        agent = self._agent(speed_mps=6.4, target_mps=6.2,
                            accel_mps2=-3.0)

        target_kmh = agent._plan_target_speed(27.0, [], 0.1)

        self.assertGreaterEqual(target_kmh / 3.6, 6.4)
        self.assertGreaterEqual(agent._last_target_accel_mps2, 0.0)
        self.assertGreater(agent._last_longitudinal_plan[
            "preview_speed_mps"], 6.4)

    def test_distant_curve_does_not_preserve_stale_braking(self):
        agent = self._agent(speed_mps=0.1, target_mps=0.0,
                            accel_mps2=-3.0)

        target_kmh = agent._plan_target_speed(27.0, [(12.0, 5.3)], 0.1)

        self.assertGreater(target_kmh, 0.0)
        self.assertGreaterEqual(agent._last_target_accel_mps2, 0.0)

    def test_stop_constraint_retains_braking_memory(self):
        agent = self._agent(speed_mps=6.4, target_mps=6.2,
                            accel_mps2=-3.0)

        agent._plan_target_speed(27.0, [(12.0, 0.0)], 0.1)

        self.assertLess(agent._last_target_accel_mps2, 0.0)

    def test_distant_stop_does_not_make_cruise_stop_now(self):
        agent = self._agent(speed_mps=3.4, target_mps=3.7,
                            accel_mps2=-3.0)

        target_kmh = agent._plan_target_speed(27.0, [(50.0, 0.0)], 0.1)

        self.assertGreaterEqual(agent._last_target_accel_mps2, 0.0)
        self.assertGreater(target_kmh / 3.6, 3.4)


if __name__ == "__main__":
    unittest.main()
