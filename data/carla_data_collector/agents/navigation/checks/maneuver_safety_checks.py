"""Receiving-lane and forecast checks for lateral maneuvers."""

import pathlib
import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
from agents.navigation.behavior_prediction import ActorPrediction
from agents.navigation.maneuver_safety import assess_maneuver, parallel_hold_path, choose_candidate, waiting_annotation
from agents.navigation.route_recovery import temporary_route


class ManeuverSafetyChecks(unittest.TestCase):
    def test_source_follower_must_leave_time_for_complete_lane_exit(self):
        # Ego is in the borrowed lane, returning toward its original route.
        route = [[float(x), 0.] for x in range(-30, 81, 2)]
        path = [[0., -3.5], [5., -3.5], [10., -2.], [15., 0.], [30., 0.]]
        actor = self.actor([-12., -3.5], (6., 0.))
        result = assess_maneuver(route, path, [actor], [0., -3.5], 2.5,
                                 (2., 1.), 0.)
        self.assertFalse(result['safe'])
        self.assertIn('source_lane_clearance', [item['kind'] for item in result['blocking_actors']])

    def test_clear_recovery_confirmation_is_not_a_predicted_conflict(self):
        safe = {'safe': True, 'ready': False, 'blocking_actors': []}
        for side in ['LEFT', 'RIGHT']:
            state, reason = 'RECOVER_' + side, 'route_recover_' + side.lower()
            self.assertEqual(waiting_annotation(safe, state, reason), (state, reason))
        self.assertEqual(waiting_annotation(safe, 'RECOVER_LEFT', 'route_recover_left', True),
                         ('YIELD', 'predicted_conflict'))

    def test_real_actor_block_and_geometry_rejection_have_different_reasons(self):
        blocked = {'safe': False, 'blocking_actors': [{'kind': 'forecast_conflict'}]}
        self.assertEqual(waiting_annotation(blocked, 'RECOVER_LEFT', 'route_recover_left'),
                         ('YIELD', 'predicted_conflict'))
        self.assertEqual(waiting_annotation({'safe': False, 'geometry': {'safe': False}},
                         'RECOVER_LEFT', 'route_recover_left'), ('YIELD', 'comfortable_braking'))
        self.assertEqual(waiting_annotation({'candidates': [blocked]}, 'FOLLOW', 'follow_vehicle'),
                         ('YIELD', 'predicted_conflict'))

    def test_stale_safe_candidate_does_not_override_current_geometry_rejection(self):
        safety = {'safe': False, 'blocking_actors': [], 'candidates': [{'safe': True}]}
        self.assertEqual(waiting_annotation(safety, 'AVOID_LEFT', 'obstacle_avoid_left'),
                         ('YIELD', 'comfortable_braking'))

    def setUp(self):
        self.route = [[float(x), 0.] for x in range(-10, 91, 2)]
        self.bypass = temporary_route(self.route, [0., 0.], 1, 3.5,
                                      obstacle_position=[25., 0.])["points"]

    def actor(self, position, velocity=(0., 0.), actor_id=1, modes=()):
        times = np.arange(.25, 3.01, .25)
        return ActorPrediction(actor_id, "vehicle", times,
            np.asarray(position) + times[:, None] * np.asarray(velocity),
            np.tile(np.eye(2)[None] * .05, (len(times), 1, 1)),
            float(np.linalg.norm(velocity)), (2., 1.), modes, tuple(position))

    def assess(self, actors, path=None, position=(0., 0.), target=3.5, speed=3.):
        return assess_maneuver(self.route, self.bypass if path is None else path,
                               actors, position, speed, (2.4, 1.08), target)

    def test_parked_obstacle_safe_bypass_is_not_blocked_by_diffuse_covariance(self):
        actor = self.actor([25., 0.])
        actor.covariances[:] = np.eye(2) * 100.
        self.assertTrue(self.assess([actor])["safe"])

    def test_parked_vehicle_on_receiving_path_blocks_entry(self):
        self.assertFalse(self.assess([self.actor([25., -3.5])])["safe"])

    def test_fast_rear_vehicle_outside_old_twelve_meter_zone_blocks(self):
        result = self.assess([self.actor([-40., -3.5], [18., 0.])])
        self.assertFalse(result["safe"])
        self.assertEqual(result["blocking_actors"][0]["kind"], "receiving_gap")

    def test_oncoming_gap_uses_sum_of_speeds(self):
        result = self.assess([self.actor([40., -3.5], [-10., 0.])], speed=3.)
        self.assertFalse(result["safe"])
        self.assertAlmostEqual(result["blocking_actors"][0]["closing_speed_mps"], 13.)

    def test_oncoming_reservation_extends_beyond_three_second_forecast(self):
        result = assess_maneuver(self.route, self.bypass,
            [self.actor([120., -3.5], [-12., 0.])], [0., 0.], 2.5,
            (2.4, 1.08), 3.5, reservation_s=15.)
        self.assertFalse(result["safe"])
        self.assertEqual(result["blocking_actors"][0]["kind"], "oncoming_reservation")

    def test_behind_opposite_mover_with_crossing_motion_not_ignored(self):
        path = [[0., 0.], [40., 0.]]
        result = self.assess([self.actor([-8., -3.], [14., 2.])],
                             path=path, position=(0., 0.), target=0., speed=3.)
        self.assertFalse(result["safe"])

    def test_same_direction_preferred_even_if_opposing_has_lower_cost(self):
        same = dict(side=-1, safe=True, ready=True, same_direction=True, cost=5.)
        opposing = dict(side=1, safe=True, ready=True, same_direction=False, cost=0.)
        self.assertIs(choose_candidate([opposing, same]), same)
        same["ready"] = False
        self.assertIsNone(choose_candidate([opposing, same]))

    def test_both_same_direction_candidates_compared_by_cost(self):
        left = dict(side=1, safe=True, ready=True, same_direction=True, cost=5.)
        right = dict(side=-1, safe=True, ready=True, same_direction=True, cost=1.)
        self.assertIs(choose_candidate([left, right]), right)
        right.update(safe=False, ready=False)
        self.assertIs(choose_candidate([left, right]), left)

    def test_recovery_waits_for_occupied_original_lane(self):
        path = temporary_route(self.route, [0., -3.5], -1, 3.5)["points"]
        self.assertFalse(self.assess([self.actor([1., 0.], [3., 0.])],
                                     path, [0., -3.5], 0.)["safe"])

    def test_same_direction_map_lane_does_not_hide_wrongway_reservation(self):
        result = assess_maneuver(self.route, self.bypass,
            [self.actor([140., -3.5], [-12., 0.])], [0., 0.], 2.5,
            (2.4, 1.08), 3.5, reservation_s=20., opposing_lane=False)
        self.assertFalse(result['safe'])
        self.assertEqual(result['blocking_actors'][0]['kind'], 'oncoming_reservation')

    def test_parallel_hold_does_not_brake_for_stable_adjacent_vehicle(self):
        path, _ = parallel_hold_path(self.route, [0., -3.5])
        self.assertTrue(self.assess([self.actor([0., 0.], [3., 0.])],
                                    path, [0., -3.5], -3.5)["safe"])

    def test_diffuse_parallel_vehicle_outside_entire_recovery_does_not_block(self):
        path = temporary_route(self.route, [0., -3.5], -1, 3.5)['points']
        actor = self.actor([0., 3.5], [3., 0.])
        actor.covariances[:, 1, 1] = 9.
        self.assertTrue(self.assess([actor], path, [0., -3.5], 0.)['safe'])
        actor.actor_type = 'pedestrian'
        self.assertFalse(self.assess([actor], path, [0., -3.5], 0.)['safe'])

    def test_rear_actor_is_not_projected_onto_backward_extended_return_tangent(self):
        path = [[0., -.9], [5., -.4], [14., 0.]]
        actor = self.actor([-8., -3.5], [3., 0.])
        actor.covariances[:] = np.eye(2) * 4.
        self.assertTrue(self.assess([actor], path, [0., -.9], 0., 2.)['safe'])

    def test_already_inside_receiving_lane_does_not_yield_to_following_forecast(self):
        path = temporary_route(self.route, [0., -.25], -1, 3.5)["points"]
        self.assertTrue(self.assess([self.actor([-15., 0.], [18., 0.])],
                                    path, [0., -.25], 0.)["safe"])

    def test_partly_entered_lane_can_finish_ahead_of_slow_stable_follower(self):
        path = temporary_route(self.route, [0., -1.2], -1, 3.5)['points']
        self.assertTrue(self.assess([self.actor([-8., 0.], [3.3, 0.])],
                                    path, [0., -1.2], 0., 3.)['safe'])
        self.assertFalse(self.assess([self.actor([-8., 0.], [18., 0.])],
                                     path, [0., -1.2], 0., 3.)['safe'])

    def test_recovery_leaves_stable_current_lane_follower_without_braking(self):
        path = temporary_route(self.route, [0., -3.1], -1, 3.5)['points']
        actor = self.actor([-8., -3.5], [3., 0.])
        actor.covariances[:] = np.eye(2) * 4.
        self.assertTrue(self.assess([actor], path, [0., -3.1], 0., 3.)['safe'])
        actor.actor_type = 'pedestrian'
        self.assertFalse(self.assess([actor], path, [0., -3.1], 0., 3.)['safe'])

    def test_leaving_current_lane_does_not_ignore_fast_or_inward_follower(self):
        path = temporary_route(self.route, [0., -3.1], -1, 3.5)['points']
        for velocity in ([18., 0.], [3., 1.]):
            actor = self.actor([-8., -3.5], velocity)
            actor.covariances[:] = np.eye(2) * 4.
            self.assertFalse(self.assess([actor], path, [0., -3.1], 0., 3.)['safe'])

    def test_distant_passed_actor_is_not_clamped_into_receiving_gap(self):
        self.route = [[float(x), 0.] for x in range(-4, 91, 2)]
        path = temporary_route(self.route, [0., -3.5], -1, 3.5)["points"]
        self.assertTrue(self.assess([self.actor([-35., 0.])], path,
                                    [0., -3.5], 0.)["safe"])

    def test_current_yaw_projection_does_not_double_inflate_future_escape_path(self):
        position = [24., -2.6]
        path = temporary_route(self.route, position, 1, 3.5,
                               obstacle_position=[25., 0.])["points"]
        result = assess_maneuver(self.route, path, [self.actor([25., 0.])],
            position, 1., (2.4, 1.08), 3.5, current_half_width_m=1.7)
        self.assertTrue(result["safe"])

    def test_intent_probabilities_not_union_of_hypotheses(self):
        actor = self.actor([0., 9.], [3., 0.])
        times = actor.times
        mode = SimpleNamespace(name="change_left", probability=.04,
            means=np.asarray([self.bypass[0]] * len(times), dtype=float),
            covariances=actor.covariances)
        # Put the hypothetical mode on ego's future path.
        from agents.navigation.route_recovery import sample_route
        mode.means = np.asarray([sample_route(self.bypass, 3.*t)[0] for t in times])
        actor.modes = (mode,)
        self.assertTrue(self.assess([actor])["safe"])
        mode.probability = .8
        self.assertFalse(self.assess([actor])["safe"])


if __name__ == "__main__":
    unittest.main()
