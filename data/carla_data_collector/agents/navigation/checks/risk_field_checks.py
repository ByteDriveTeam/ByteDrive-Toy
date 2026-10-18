"""Regression checks for route risk directionality."""

import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
from agents.navigation.behavior_prediction import ActorPrediction
from agents.navigation.risk_field import evaluate_route_risk


class RouteRiskDirectionalityChecks(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
