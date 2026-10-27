"""Behavior checks for deterministic multi-intent actor forecasts."""

import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
from agents.navigation.behavior_prediction import ActorPrediction, _Track
from agents.navigation.intent_prediction import IntentMode, motion_lane_context, predict_intent_modes
from agents.navigation.risk_field import (
    evaluate_route_risk, plan_front_vehicle_speed)


class IntentPredictionChecks(unittest.TestCase):
    def test_stopped_vehicle_is_not_extrapolated_backwards(self):
        track = _Track(1, 'vehicle', 1.5, .25, (2.4, .9))
        for index in range(7):
            t = index * .1
            move_t = min(t, .5)
            track.update([2. * move_t - 2. * move_t**2, 0.],
                         [max(2. - 4. * t, 0.), 0.], t)
        forecast = track.predict(4., .25)
        np.testing.assert_allclose(forecast.means, np.tile([[.5, 0.]], (16, 1)), atol=1e-6)

    def test_finished_lateral_cut_does_not_predict_a_reverse_cut(self):
        track = _Track(1, 'vehicle', 1.5, .25, (2.4, .9))
        context = dict(direction=np.array([1., 0.]), left=True, right=True, offset_m=0.)
        for index in range(9):
            t = index * .1
            track.update([2. * t, min(t, .6)], [2., 1. if t < .6 else 0.], t, context)
        np.testing.assert_allclose(track.predict(3., .25).means[:, 1], .6, atol=1e-6)

    def test_velocity_tick_oscillation_does_not_create_sustained_acceleration(self):
        track = _Track(1, 'vehicle', 1.5, .25, (2.4, .9))
        for index in range(60):
            track.update([index * .4, 0.], [4. + (-1)**index * .17, 0.], index * .1)
        self.assertLess(float(np.linalg.norm(track.x[4:6])), .01)
        self.assertAlmostEqual(track.last_observed_velocity[0], 3.83)

    def test_wrong_way_intents_keep_observed_direction(self):
        context = {"direction": np.asarray([1., 0.]), "left": True,
                   "right": False, "offset_m": .4}
        oriented = motion_lane_context(context, [-8., 0.])
        self.assertTrue(oriented["wrong_way"])
        self.assertFalse(oriented["left"])
        self.assertTrue(oriented["right"])
        np.testing.assert_array_equal(context["direction"], [1., 0.])
        times = np.arange(.25, 3.01, .25)
        modes = predict_intent_modes([30., 0.], [-8., 0.], times,
            np.tile(np.eye(2)[None], (len(times), 1, 1)),
            {"keep": .8, "brake": .2}, context, 0.)
        keep = next(mode for mode in modes if mode.name == "keep")
        self.assertAlmostEqual(float(keep.means[-1, 0]), 6., places=4)
        self.assertLess(float(modes[1].means[-1, 0]), 30.)

    def test_crossing_motion_is_not_forced_to_map_heading(self):
        context = {"direction": np.asarray([1., 0.])}
        self.assertIsNone(motion_lane_context(context, [0., 6.]))

    def setUp(self):
        self.lane = {"direction": np.asarray([1.0, 0.0]),
                     "offset_m": 0.0, "lane_width_m": 3.5,
                     "left": True, "right": True}

    def _track(self, observations):
        track = _Track(1, "vehicle", 1.5, 0.25, (2.4, 0.9))
        for index, (x, y, vx, vy) in enumerate(observations):
            track.update([x, y], [vx, vy], index * 0.1,
                         lane_context=self.lane)
        return track

    def test_steady_vehicle_keeps_lane_without_false_side_intent(self):
        track = self._track([(0.6 * i, 0.0, 6.0, 0.0)
                             for i in range(12)])
        prediction = track.predict(3.0, 0.25)
        probabilities = {mode.name: mode.probability
                         for mode in prediction.modes}

        self.assertAlmostEqual(sum(probabilities.values()), 1.0)
        self.assertGreater(probabilities["keep"], 0.75)
        self.assertLess(probabilities["change_left"], 0.05)
        self.assertLess(probabilities["change_right"], 0.05)

    def test_observed_velocity_is_not_replaced_by_filtered_future_velocity(self):
        track = self._track([(0., 0., 4., 0.), (.4, 0., 0., 0.)])
        forecast = track.predict(3., .25)
        np.testing.assert_array_equal(forecast.current_velocity, [0., 0.])
        np.testing.assert_allclose(forecast.means[:, 0], .4, atol=1e-6)

    def test_history_shifts_brake_and_lane_change_probabilities(self):
        braking = self._track([
            (0.8 * i - 0.01 * i * i, 0.0, 8.0 - 0.2 * i, 0.0)
            for i in range(10)])
        brake_probs = {mode.name: mode.probability
                       for mode in braking.predict(3.0, 0.25).modes}
        self.assertGreater(brake_probs["brake"], brake_probs["keep"])

        changing = self._track([(0.6 * i, 0.12 * i, 6.0, 1.2)
                                for i in range(10)])
        forecast = changing.predict(3.0, 0.25)
        probabilities = {mode.name: mode.probability
                         for mode in forecast.modes}
        self.assertGreater(probabilities["change_left"], 0.5)
        self.assertLess(probabilities["change_right"], 0.05)
        left = next(mode for mode in forecast.modes
                    if mode.name == "change_left")
        keep = next(mode for mode in forecast.modes
                    if mode.name == "keep")
        self.assertGreater(left.means[-1, 1] - keep.means[-1, 1], 2.0)

    def test_prediction_is_reproducible_without_sampling(self):
        observations = [(0.6 * i, 0.08 * i, 6.0, 0.8)
                        for i in range(10)]
        first = self._track(observations).predict(3.0, 0.25)
        second = self._track(observations).predict(3.0, 0.25)
        for left, right in zip(first.modes, second.modes):
            self.assertEqual(left.name, right.name)
            self.assertEqual(left.probability, right.probability)
            np.testing.assert_array_equal(left.means, right.means)

    def test_feature_switch_preserves_single_track_prediction(self):
        track = self._track([(0.6 * i, 0.0, 6.0, 0.0)
                             for i in range(6)])
        prediction = track.predict(
            3.0, 0.25, multi_intent_enabled=False)
        self.assertEqual(prediction.modes, ())
        self.assertEqual(prediction.means.shape, (12, 2))


class IntentRiskChecks(unittest.TestCase):
    def setUp(self):
        self.route = [[float(x), 0.0] for x in range(0, 61, 5)]
        self.times = np.arange(0.25, 3.01, 0.25)
        self.covariances = np.repeat(
            (np.eye(2) * 0.1)[None], len(self.times), axis=0)

    def _prediction(self, actor_id, base, modes):
        return ActorPrediction(
            actor_id=actor_id, actor_type="vehicle", times=self.times,
            means=np.asarray(base), covariances=self.covariances,
            speed_mps=6.0, extent_xy=(2.4, 0.9), modes=tuple(
                IntentMode(name=name, probability=probability,
                           means=np.asarray(points),
                           covariances=self.covariances,
                           speed_mps=6.0)
                for name, probability, points in modes))

    def _front(self, predictions):
        return plan_front_vehicle_speed(
            self.route, predictions, [0, 0], 10.0,
            2.4, 0.92, 0.35, 0.7, 3.0)

    def test_braking_intent_lowers_front_speed_before_immediate_hazard(self):
        keep = [[25.0 + 6.0 * t, 0.0] for t in self.times]
        brake = [[25.0 + 6.0 * t - t * t, 0.0] for t in self.times]
        baseline = self._prediction(10, keep, [])
        mixture = self._prediction(10, keep, [
            ("keep", 0.6, keep), ("brake", 0.4, brake)])

        plan = self._front([mixture])

        self.assertLess(plan["speed_cap_mps"],
                        self._front([baseline])["speed_cap_mps"])
        self.assertFalse(plan["emergency"])

    def test_low_probability_side_intent_does_not_brake_parallel_ego(self):
        keep = [[14.0 + 6.0 * t, 3.5] for t in self.times]
        cut_in = [[14.0 + 6.0 * t, 3.5 * (1.0 - min(t / 3.0, 1.0))]
                  for t in self.times]
        low = self._prediction(11, keep, [
            ("keep", 0.95, keep), ("change_right", 0.05, cut_in)])
        high = self._prediction(11, keep, [
            ("keep", 0.4, keep), ("change_right", 0.6, cut_in)])

        self.assertGreaterEqual(self._front([low])["speed_cap_mps"], 10.0)
        self.assertFalse(self._front([low])["emergency"])
        self.assertFalse(self._front([low]).get("crossing_constraint", True))
        self.assertLess(self._front([high])["speed_cap_mps"],
                        self._front([low])["speed_cap_mps"])
        self.assertTrue(self._front([high]).get("crossing_constraint", False))
        self.assertGreater(evaluate_route_risk(
            self.route, [high], [0, 0], 10.0)["max_risk"],
            evaluate_route_risk(
                self.route, [low], [0, 0], 10.0)["max_risk"])


if __name__ == "__main__":
    unittest.main()
