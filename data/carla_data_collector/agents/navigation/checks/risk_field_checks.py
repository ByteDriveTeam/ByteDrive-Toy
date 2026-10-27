"""Regression checks for route risk directionality."""

import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
from agents.navigation.behavior_prediction import ActorPrediction
from agents.navigation.risk_field import (
    evaluate_route_risk, plan_front_vehicle_speed)


class RouteRiskDirectionalityChecks(unittest.TestCase):
    def test_submillimetre_origin_jitter_does_not_create_a_forward_rear_actor(self):
        route = [[0., 0.], [0., .000008], [0., -2.], [0., -40.]]
        rear = ActorPrediction(77, 'vehicle', np.array([.25, .5, 1.]),
            np.repeat(np.array([[-3.7, 27.]]), 3, axis=0),
            np.repeat((np.eye(2) * .1)[None], 3, axis=0), 0., (2.4, .9),
            current_position=(-3.7, 27.), current_velocity=(0., 0.))
        result = plan_front_vehicle_speed(route, [rear], [0., 0.], 0.,
                                          2.4, 1.08, .35, .7, 3.)
        self.assertIsNone(result['kind'])
        rear.current_position = (0., -15.)
        rear.means[:] = [0., -15.]
        self.assertEqual(plan_front_vehicle_speed(route, [rear], [0., 0.], 2.,
            2.4, 1.08, .35, .7, 3.)['kind'], 'following')

    def setUp(self):
        self.route = [[0.0, 0.0], [5.0, 0.0], [10.0, 0.0], [15.0, 0.0]]
        self.times = np.asarray([0.25, 0.5, 1.0, 1.5, 2.0, 3.0])
        self.covariances = np.repeat(
            (np.eye(2) * 0.1)[None], len(self.times), axis=0)

    def prediction(self, actor_id, positions):
        return ActorPrediction(
            actor_id=actor_id, actor_type="vehicle", times=self.times,
            means=np.asarray(positions, dtype=np.float64),
            covariances=self.covariances, speed_mps=3.0,
            extent_xy=(2.4, 0.9))

    def risk(self, predictions):
        return evaluate_route_risk(
            self.route, predictions, [0.0, 0.0], 3.0)

    def test_rear_actor_accelerating_into_forecast_does_not_brake_ego(self):
        rear = self.prediction(1, [
            [-8, 0], [-7, 0], [-5, 0], [-2, 0], [2, 0], [6, 0]])
        result = self.risk([rear])
        self.assertEqual(result["max_risk"], 0.0)
        self.assertIsNone(result["closest_actor_id"])

    def test_front_vehicle_and_crossing_vehicle_remain_hazards(self):
        front = self.prediction(2, [[6, 0]] * len(self.times))
        crossing = self.prediction(3, [
            [6, 5], [6, 4], [6, 2], [6, 0], [6, -1], [6, -2]])
        for actor in (front, crossing):
            with self.subTest(actor_id=actor.actor_id):
                result = self.risk([actor])
                self.assertGreater(result["max_risk"], 0.42)
                self.assertEqual(result["closest_actor_id"], actor.actor_id)

    def test_pedestrian_crossing_from_rear_side_remains_a_hazard(self):
        crossing = self.prediction(4, [
            [-3, 2], [-2, 2], [0, 1], [2, 0], [2, 0], [2, 0]])
        crossing.actor_type = "pedestrian"
        crossing.extent_xy = (0.3, 0.3)
        result = evaluate_route_risk(
            self.route, [crossing], [0.0, 0.0], 0.0)
        self.assertGreater(result["max_risk"], 0.42)
        self.assertEqual(result["closest_actor_id"], crossing.actor_id)

    def test_stable_adjacent_vehicle_does_not_cause_braking(self):
        for side in (-1.0, 1.0):
            with self.subTest(side=side):
                adjacent = self.prediction(5, [[6, side * 3.5]] * len(self.times))
                adjacent.extent_xy = (5.1, 1.97)
                result = evaluate_route_risk(
                    self.route, [adjacent], [0.0, 0.0], 3.0,
                    route_width_m=1.27)
                self.assertLess(result["max_risk"], 0.18)

    def test_adjacent_vehicle_cutting_in_remains_a_hazard(self):
        cutting_in = self.prediction(6, [
            [8, 3.5], [8, 3.0], [8, 2.4], [8, 1.5], [8, 0.5], [8, 0]])
        cutting_in.extent_xy = (5.1, 1.97)
        result = evaluate_route_risk(
            self.route, [cutting_in], [0.0, 0.0], 3.0,
            route_width_m=1.27)
        self.assertGreater(result["max_risk"], 0.42)
        self.assertEqual(result["closest_actor_id"], cutting_in.actor_id)

    def test_vehicle_between_sparse_route_points_keeps_full_risk(self):
        centered = self.prediction(8, [[2.5, 0.0]] * len(self.times))
        result = evaluate_route_risk(
            self.route, [centered], [0.0, 0.0], 0.0,
            route_width_m=1.27)
        self.assertGreater(result["max_risk"], 0.42)

    def test_pedestrian_keeps_legacy_roadside_corridor(self):
        roadside = self.prediction(7, [[1, 1.8]] * len(self.times))
        roadside.actor_type = "pedestrian"
        roadside.extent_xy = (0.3, 0.3)
        result = evaluate_route_risk(
            self.route, [roadside], [0.0, 0.0], 0.0,
            route_width_m=1.27, pedestrian_route_width_m=1.925)
        self.assertGreater(result["max_risk"], 0.42)

    def test_front_plan_brakes_before_three_second_risk_overlap(self):
        slow = self.prediction(9, [[25 + 6 * time_s, 0]
                                   for time_s in self.times])
        slow.speed_mps = 6.0
        route = [[x, 0] for x in range(0, 61, 5)]
        risk = evaluate_route_risk(route, [slow], [0, 0], 12.0,
                                   route_width_m=1.27)
        plan = plan_front_vehicle_speed(
            route, [slow], [0, 0], 12.0, 2.4, 0.92, 0.35, 0.7, 3.0)
        self.assertLess(risk["max_risk"], 0.18)
        self.assertLess(plan["speed_cap_mps"], 12.0)
        self.assertEqual(plan["actor_id"], slow.actor_id)
        self.assertFalse(plan["emergency"])

    def test_front_plan_keeps_parallel_and_rear_vehicles_out(self):
        route = [[x, 0] for x in range(0, 61, 5)]
        adjacent = self.prediction(10, [[25 + 6 * time_s, 3.5]
                                       for time_s in self.times])
        adjacent.speed_mps = 6.0
        adjacent.extent_xy = (5.1, 1.97)
        rear = self.prediction(11, [[-8 + 6 * time_s, 0]
                                   for time_s in self.times])
        rear.speed_mps = 6.0
        plan = plan_front_vehicle_speed(
            route, [adjacent, rear], [0, 0], 12.0, 2.4, 0.92,
            0.35, 0.7, 3.0)
        self.assertEqual(plan["speed_cap_mps"], float("inf"))
        self.assertIsNone(plan["actor_id"])

    def test_front_plan_does_not_emergency_brake_for_long_parallel_bus(self):
        times = np.arange(0.25, 3.01, 0.25)
        bus = self.prediction(15, [[7.0 + 5.0 * time_s, 3.5]
                                   for time_s in times])
        bus.times = times
        bus.covariances = np.repeat(
            (np.eye(2) * 0.1)[None], len(times), axis=0)
        bus.speed_mps = 5.0
        bus.extent_xy = (5.1, 1.97)
        route = [[0, 0], [5, 0], [10, 0], [15, 0], [20, 0.5]]
        plan = plan_front_vehicle_speed(
            route, [bus], [0, 0], 7.6, 2.4, 0.92, 0.35, 0.7, 3.0)
        self.assertIsNone(plan["actor_id"])
        self.assertFalse(plan["emergency"])

    def test_front_plan_ignores_rear_vehicle_forecast_near_bending_route(self):
        times = np.arange(0.25, 3.01, 0.25)
        rear = self.prediction(16, [
            [10.9, 2.5], [12.0, 2.5], [13.3, 2.5], [14.7, 2.5],
            [16.4, 2.5], [18.2, 2.5], [20.2, 2.5], [22.4, 2.5],
            [24.8, 2.5], [27.3, 2.5], [30.1, 2.5], [33.0, 2.5]])
        rear.times = times
        rear.covariances = np.repeat(
            (np.eye(2) * 0.1)[None], len(times), axis=0)
        rear.extent_xy = (2.3, 0.97)
        route = [[23.16, 1.17], [25.7, -0.1], [27.4, -1.7],
                 [28.8, -3.5], [29.9, -5.4], [30.7, -7.5]]
        plan = plan_front_vehicle_speed(
            route, [rear], route[0], 3.3, 2.4, 0.92, 0.35, 0.7, 3.0)
        self.assertIsNone(plan["actor_id"])
        self.assertFalse(plan["emergency"])

    def test_front_plan_emergency_for_close_stopped_vehicle(self):
        stopped = self.prediction(12, [[10, 0]] * len(self.times))
        stopped.speed_mps = 0.0
        plan = plan_front_vehicle_speed(
            self.route, [stopped], [0, 0], 12.0, 2.4, 0.92,
            0.35, 0.7, 3.0)
        self.assertTrue(plan["emergency"])
        self.assertLess(plan["speed_cap_mps"], 12.0)

    def test_front_plan_slows_for_uncertain_crossing_before_risk_threshold(self):
        route = [[x, 0] for x in range(0, 61, 5)]
        crossing = self.prediction(13, [
            [15.2, 14.26], [15.2, 13.27], [15.2, 12.07],
            [15.2, 10.64], [15.2, 8.99], [15.2, 7.13],
            [15.2, 5.04], [15.2, 2.74], [15.2, 0.21],
            [15.2, -2.53], [15.2, -5.49], [15.2, -8.68]])
        crossing.times = np.arange(0.25, 3.01, 0.25)
        crossing.covariances = np.asarray([
            np.eye(2) * (0.1 + 13.05 * (time_s / 2.25) ** 4)
            for time_s in crossing.times])
        plan = plan_front_vehicle_speed(
            route, [crossing], [0, 0], 5.2, 2.4, 0.92,
            0.35, 0.7, 3.0)
        self.assertEqual(plan["actor_id"], crossing.actor_id)
        self.assertEqual(plan["kind"], "crossing")
        self.assertLess(plan["speed_cap_mps"], 5.2)

    def test_front_plan_stops_for_imminent_crossing(self):
        crossing = self.prediction(14, [
            [11.0, 11.58], [11.0, 9.89], [11.0, 7.94],
            [11.0, 5.71], [11.0, 3.21], [11.0, 0.43],
            [11.0, -2.62], [11.0, -5.95], [11.0, -9.55],
            [11.0, -13.43], [11.0, -17.58], [11.0, -22.01]])
        crossing.times = np.arange(0.25, 3.01, 0.25)
        crossing.covariances = np.repeat(
            (np.eye(2) * 0.5)[None], len(crossing.times), axis=0)
        route = [[x, 0] for x in range(0, 61, 5)]
        plan = plan_front_vehicle_speed(
            route, [crossing], [0, 0], 6.9, 2.4, 0.92,
            0.35, 0.7, 3.0)
        self.assertEqual(plan["kind"], "crossing")
        self.assertTrue(plan["emergency"])


if __name__ == "__main__":
    unittest.main()
