"""Defensive reserves must not regress the rear/parallel-traffic exclusions."""
import pathlib
import sys
import unittest
from types import SimpleNamespace
import numpy as np
import carla

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
from agents.navigation.defensive_driving import DefensivePlanner, reserve_distance, reserve_speed
from agents.navigation.defensive_visibility import defensive_road_constraints, JunctionVisibility
from agents.navigation.behavior_prediction import ActorPrediction
from agents.navigation.intent_prediction import IntentMode


def actor(position=(25., 0.), velocity=(6., 0.), actor_id=2):
    times = np.arange(.25, 3.01, .25)
    means = np.array(position)[None] + times[:, None]*np.array(velocity)[None]
    return ActorPrediction(actor_id, 'vehicle', times, means, np.tile(np.eye(2)*.1, (len(times), 1, 1)),
        float(np.linalg.norm(velocity)), (2., .9), current_position=position,
        current_velocity=velocity, observation_count=8, current_heading_xy=(1., 0.))


def update(planner, actors=(), time=0., **kwargs):
    return planner.update([[0., 0.], [120., 0.]], actors, [0., 0.], 10., (2., .9), time, 14., **kwargs)


class DefensiveDrivingChecks(unittest.TestCase):
    def test_braking_reserve_inversion_and_lead_emergency_deceleration(self):
        for speed in (3., 7., 14.):
            distance = reserve_distance(speed, 5., 1., 2.2, 3., 6., 3.)
            recovered = reserve_speed(distance, 5., 1., 2.2, 3., 6., 3.)
            self.assertAlmostEqual(recovered, speed)
        self.assertGreater(reserve_distance(10., 5., 1., 2.2, 3., 8., 3.),
                           reserve_distance(10., 5., 1., 2.2, 3., 4., 3.))

    def test_same_lane_vehicle_gets_preemptive_reserve_before_emergency(self):
        result = update(DefensivePlanner(), [actor()])
        self.assertTrue(result['active'])
        self.assertLess(result['speed_cap_mps'], 10.)
        self.assertEqual(result['kind'], 'lead_reserve')
        self.assertGreater(result['candidates'][0]['gap_m'], 15.)

    def test_stable_parallel_side_by_side_rear_and_oncoming_have_no_advisory(self):
        actors = [actor((10., 3.5)), actor((0., -3.5)), actor((-10., 0.)), actor((20., 0.), (-5., 0.))]
        for prediction in actors:
            prediction.covariances *= 1000.
        self.assertFalse(update(DefensivePlanner(), actors)['active'])

    def test_actual_inward_motion_yields_but_outward_motion_does_not(self):
        result = update(DefensivePlanner(), [actor((16., 3.5), (4., -1.))])
        self.assertEqual(result['kind'], 'cut_in_reserve')
        self.assertGreater(result['candidates'][0]['entry_time_s'], 0.)
        self.assertFalse(update(DefensivePlanner(), [actor((16., 3.5), (4., 1.))])['active'])

    def test_weak_lane_change_prior_without_inward_evidence_is_not_a_hazard(self):
        prediction = actor((16., 3.5), (4., 0.))
        means = prediction.means.copy()
        means[:, 1] -= np.linspace(0., 3.5, len(means))
        prediction.modes = (IntentMode('change_right', .04, means, prediction.covariances, 4.),)
        self.assertFalse(update(DefensivePlanner(), [prediction])['active'])
        prediction.modes = (IntentMode('change_right', .6, means, prediction.covariances, 4.),)
        self.assertTrue(update(DefensivePlanner(), [prediction])['active'])

    def test_supported_braking_intent_increases_reserve(self):
        prediction = actor()
        ordinary = update(DefensivePlanner(), [prediction])['speed_cap_mps']
        means = prediction.means.copy()
        means[:, 0] -= np.minimum(prediction.times**2, 5.)
        prediction.modes = (IntentMode('brake', .6, means, prediction.covariances, 6.),)
        self.assertLess(update(DefensivePlanner(), [prediction])['speed_cap_mps'], ordinary)

    def test_stopped_lead_does_not_relaunch_after_a_physics_speed_drop(self):
        planner = DefensivePlanner()
        prediction = actor((12., 0.), (0., 0.))
        first = update(planner, [prediction])
        self.assertTrue(first['stopping_committed'])
        slow = planner.update([[0., 0.], [120., 0.]], [prediction], [0., 0.], .7,
                              (2., .9), .1, 14.)
        self.assertLessEqual(slow['speed_cap_mps'], .7)
        self.assertTrue(slow['stopping_committed'])
        stopped = planner.update([[0., 0.], [120., 0.]], [prediction], [0., 0.], 0.,
                                 (2., .9), .2, 14.)
        self.assertEqual(stopped['speed_cap_mps'], 0.)
        moving = update(planner, [actor((12., 0.), (2., 0.))], .3)
        self.assertFalse(moving['stopping_committed'])
        self.assertGreater(moving['speed_cap_mps'], 0.)

    def test_stationary_side_actor_and_slow_moving_lead_do_not_latch_a_stop(self):
        self.assertFalse(update(DefensivePlanner(), [actor((12., 3.5), (0., 0.))])['stopping_committed'])
        self.assertFalse(update(DefensivePlanner(), [actor((12., 0.), (1., 0.))])['stopping_committed'])

    def test_creeping_lead_releases_stop_when_observed_gap_opens(self):
        planner = DefensivePlanner()
        update(planner, [actor((12., 0.), (0., 0.))])
        planner.update([[0., 0.], [120., 0.]], [actor((12., 0.), (0., 0.))],
                       [0., 0.], 0., (2., .9), .1, 14.)
        result = planner.update([[0., 0.], [120., 0.]], [actor((14., 0.), (.2, 0.))],
                                [0., 0.], 0., (2., .9), .2, 14.)
        self.assertFalse(result['stopping_committed'])
        self.assertGreater(result['speed_cap_mps'], 0.)

    def test_release_hysteresis_rate_and_immediate_tightening(self):
        planner = DefensivePlanner()
        first = update(planner, [actor()])['speed_cap_mps']
        self.assertEqual(update(planner, [], .1)['speed_cap_mps'], first)
        self.assertEqual(update(planner, [], .5)['speed_cap_mps'], first)
        released = update(planner, [], .8)['speed_cap_mps']
        self.assertGreater(released, first)
        self.assertLessEqual(released-first, .8*.25 + 1e-8)
        self.assertLess(update(planner, [actor((10., 0.))], .9)['speed_cap_mps'], first)

    def test_disabled_suspension_and_time_rewind_remove_stale_state(self):
        self.assertFalse(update(DefensivePlanner({'defensive_enabled': False}), [actor()])['active'])
        planner = DefensivePlanner()
        update(planner, [actor()], 10.)
        self.assertFalse(update(planner, [], 10.1, suspended=True)['active'])
        update(planner, [actor()], 10.2)
        self.assertFalse(update(planner, [], 1.)['active'])

    def test_blind_junction_only_adds_limit_when_sight_is_obstructed(self):
        blind = dict(occluded=True, distance_m=20., speed_limit_mps=2.5)
        constraints, reasons = defensive_road_constraints({'points': []}, blind, 5., {})
        self.assertEqual(constraints, [(15., 2.5)])
        self.assertEqual(reasons[0]['kind'], 'blind_junction')
        self.assertEqual(defensive_road_constraints({'points': []}, dict(blind, occluded=False), 5., {}), ([], []))

    def test_turn_reserve_is_mirrored_and_appears_before_turn(self):
        for sign in (-1, 1):
            points = [dict(arc_m=i*3.+15., location_world=p) for i, p in enumerate(
                ([0., 0.], [3., 0.], [5., sign*2.], [5., sign*5.]))]
            constraints, reasons = defensive_road_constraints({'points': points}, {}, 4., {})
            self.assertTrue(constraints)
            self.assertTrue(all(distance > 0. and limit < 6. for distance, limit in constraints))
            self.assertTrue(all(reason['kind'] == 'turn_reserve' for reason in reasons))

    def test_real_ray_query_outcome_and_cache(self):
        entry = SimpleNamespace(road_id=2, section_id=0, lane_id=-1,
            transform=carla.Transform(carla.Location(x=20., y=5.), carla.Rotation(yaw=-90.)),
            previous=lambda distance: [SimpleNamespace(transform=carla.Transform(carla.Location(x=20., y=25.)))])
        junction = SimpleNamespace(id=3, get_waypoints=lambda _: [(entry, entry)])
        waypoint = SimpleNamespace(is_junction=True, get_junction=lambda: junction,
            transform=carla.Transform(carla.Location(x=20.)))
        world = SimpleNamespace(cast_ray=lambda start, end: [SimpleNamespace(location=carla.Location(x=10., y=5.))])
        planner = JunctionVisibility()
        result = planner.update(world, [(waypoint, 4)], carla.Transform(), SimpleNamespace(x=2.), 1.)
        self.assertTrue(result['checked'] and result['occluded'])
        world.cast_ray = lambda *args: []
        self.assertTrue(planner.update(world, [(waypoint, 4)], carla.Transform(), SimpleNamespace(x=2.), 1.1)['occluded'])
        self.assertFalse(planner.update(world, [(waypoint, 4)], carla.Transform(), SimpleNamespace(x=2.), 1.5)['occluded'])


if __name__ == '__main__':
    unittest.main()
