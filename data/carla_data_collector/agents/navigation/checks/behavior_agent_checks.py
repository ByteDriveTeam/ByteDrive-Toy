"""Regression checks for BehaviorAgent obstacle classification."""

import pathlib
import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.local_planner import RoadOption


class PedestrianAvoidanceChecks(unittest.TestCase):
    def test_navigation_goal_distance_follows_route_instead_of_shortcut(self):
        import carla
        agent = BehaviorAgent.__new__(BehaviorAgent)
        agent._vehicle = SimpleNamespace(get_location=lambda: carla.Location())
        points = [(0.,0.),(0.,20.),(5.,20.),(5.,0.)]
        plan = [(SimpleNamespace(transform=carla.Transform(carla.Location(x=x,y=y))), RoadOption.LANEFOLLOW)
                for x,y in points]
        agent._local_planner = SimpleNamespace(_stop_waypoint_creation=True,get_plan=lambda:plan)
        self.assertAlmostEqual(agent._navigation_goal_distance(),45.)
        agent._local_planner._stop_waypoint_creation=False
        self.assertIsNone(agent._navigation_goal_distance())

    def test_navigation_goal_distance_includes_gap_before_queue_start(self):
        import carla
        agent = BehaviorAgent.__new__(BehaviorAgent)
        agent._vehicle=SimpleNamespace(get_location=lambda:carla.Location())
        plan=[(SimpleNamespace(transform=carla.Transform(carla.Location(x=x))),RoadOption.LANEFOLLOW)
              for x in [4.,10.]]
        agent._local_planner=SimpleNamespace(_stop_waypoint_creation=True,get_plan=lambda:plan)
        self.assertAlmostEqual(agent._navigation_goal_distance(),10.)

    def test_committed_partial_recovery_resumes_only_if_current_checks_are_safe(self):
        import carla
        agent = BehaviorAgent.__new__(BehaviorAgent)
        waypoint = SimpleNamespace(transform=carla.Transform(), lane_width=3.5,
            get_left_lane=lambda: None, get_right_lane=lambda: None)
        agent._vehicle = SimpleNamespace(bounding_box=SimpleNamespace(
            extent=carla.Vector3D(x=2.4,y=1.08,z=.5)),
            get_transform=lambda: carla.Transform(carla.Location(y=-1.)),
            get_location=lambda: carla.Location(y=-1.))
        agent._map = SimpleNamespace(get_waypoint=lambda _: waypoint)
        agent._local_planner = SimpleNamespace(get_plan=lambda: [(waypoint, RoadOption.LANEFOLLOW)])
        for name, value in {'_speed': 3.6, '_maneuver_speed_mps': 2.5,
            '_maneuver_return_buffer_s': 4., '_last_predictions': [],
            '_maneuver_prediction_s': 3., '_maneuver_min_gap_m': 3.,
            '_maneuver_rear_ttc_s': 4., '_maneuver_conflict_probability': .2,
            '_risk_lateral_margin_m': .35, '_maneuver_oncoming_speed_mps': 15.,
            '_maneuver_max_curvature': .18, '_maneuver_last_check': {},
            '_maneuver_clear_since': {}, '_maneuver_clear_time_s': .5}.items():
            setattr(agent,name,value)
        path = {'phase': 'recover', 'points': [[0.,-1.],[14.,0.]], 'return_distance_m':14.}
        with patch('agents.navigation.behavior_agent.check_path_geometry',
                   return_value={'safe':True,'max_curvature':0.}), patch(
                   'agents.navigation.behavior_agent.assess_maneuver') as assess:
            for committed, safe, expected in [(False,True,False),(True,True,True),(True,False,False)]:
                agent._recovery_committed = committed
                assess.return_value={'safe':safe,'max_conflict_probability':0.,'min_gap_m':None}
                result=agent._maneuver_gap([[0.,0.],[50.,0.]],path,[0.,-1.],0.,('recover',-1,None),0.)
                self.assertEqual(result['ready'], expected)
                json.dumps(result)  # NumPy boolean decisions must be plain JSON booleans.

    def test_vehicle_detector_distance_is_normalized_to_centers_once(self):
        import carla
        origin = carla.Location(x=0., y=0., z=0.)
        target = SimpleNamespace(id=2, get_location=lambda: carla.Location(x=8., y=0., z=0.))
        agent = BehaviorAgent.__new__(BehaviorAgent)
        agent._vehicle = SimpleNamespace(id=1, get_location=lambda: origin)
        agent._world = SimpleNamespace(get_actors=lambda: SimpleNamespace(filter=lambda _: [target]))
        agent._direction = RoadOption.LANEFOLLOW
        agent._behavior = SimpleNamespace(min_proximity_threshold=10.)
        agent._speed_limit = 30.
        agent._vehicle_obstacle_detected = Mock(return_value=(True, target, 3.6))
        state, actor, distance = agent.collision_and_car_avoid_manager(
            SimpleNamespace(transform=SimpleNamespace(location=origin)))
        self.assertTrue(state)
        self.assertIs(actor, target)
        self.assertAlmostEqual(distance, 8.)

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
    def test_constant_low_cap_does_not_run_away_with_delayed_tracking(self):
        agent = self._agent()
        actual_speed = 0.
        peak = 0.
        for _ in range(150):
            agent._speed = actual_speed * 3.6
            command = agent._plan_target_speed(9., [], .1) / 3.6
            actual_speed += .3 * (command - actual_speed)
            peak = max(peak, actual_speed, command)
        self.assertLess(peak, 3.)
        self.assertAlmostEqual(actual_speed, 2.5, delta=.25)

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
