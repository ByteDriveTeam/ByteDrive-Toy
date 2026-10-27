"""Moving hazard admission: both triggering and time-dependent rejection."""
import pathlib
import sys
import unittest
from types import SimpleNamespace
from collections import deque
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
from agents.navigation.behavior_prediction import ActorPrediction
from agents.navigation.moving_conflicts import moving_threats, swept_motion_check, pedestrian_blocks_route
from agents.navigation.local_planner import LocalPlanner, RoadOption
from agents.navigation.route_recovery import moving_escape_route, project_route
from agents.navigation.maneuver_tracking import pursuit_steering, wheelbase_from_positions


def prediction(kind='vehicle', position=(30., 0.), velocity=(-2., 0.), actor_id=1):
    times = np.arange(.1, 4.01, .1)
    means = np.array(position)[None] + times[:, None] * np.array(velocity)[None]
    cov = np.tile(np.eye(2)[None] * .02, (len(times), 1, 1))
    return ActorPrediction(actor_id, kind, times, means, cov, float(np.linalg.norm(velocity)),
                           (2., .9) if kind == 'vehicle' else (.3, .3),
                           current_position=position, current_velocity=velocity)


class MovingConflictChecks(unittest.TestCase):
    def test_wheelbase_is_independent_of_world_heading(self):
        for yaw in (0., np.pi * .5, np.pi * .77):
            rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            points = np.array([[140., -80.], [140., 80.], [-140., -80.], [-140., 80.]]) @ rotation.T + [10000., 5000.]
            self.assertAlmostEqual(wheelbase_from_positions(points), 2.8)

    def test_escape_enters_early_but_return_clears_possible_stopping_position(self):
        path = moving_escape_route([[0., 0.], [100., 0.]], [0., 0.], 1, 3., [35., 0.], 0.)
        offsets = {round(point[0]): -point[1] for point in path['points']}
        self.assertAlmostEqual(offsets[14], 3.)
        self.assertAlmostEqual(offsets[42], 3.)
        self.assertAlmostEqual(offsets[56], 0.)
        self.assertTrue(path['return_complete'])
        short = moving_escape_route([[0., 0.], [45., 0.]], [0., 0.], 1, 3., [35., 0.], 0.)
        self.assertFalse(short['return_complete'])

    def test_tracking_mirrors_and_closed_loop_follows_escape_without_outer_excursion(self):
        for side in (-1, 1):
            path = moving_escape_route([[0., 0.], [100., 0.]], [0., 0.], side, 3., [25., 0.], 0.)['points']
            position = np.array([0., 0.]); yaw, steer = 0., 0.
            offsets = []
            for _ in range(180):
                steer = pursuit_steering(path, position, yaw, 3., 2.875, 1.22, .95, steer)
                yaw += 3. / 2.875 * np.tan(steer * 1.22 * .95) * .1
                position += np.array([np.cos(yaw), np.sin(yaw)]) * .3
                offsets.append(-position[1] * side)
            self.assertGreater(max(offsets), 2.8)
            self.assertLess(max(offsets), 3.15)
            self.assertLess(abs(position[1]), .4)

    def test_oncoming_triggers_without_stationary_wait_but_parallel_does_not(self):
        route = [[0., 0.], [100., 0.]]
        self.assertTrue(moving_threats(route, [prediction(position=(25., 0.))], [0., 0.], 3., (2., .9)))
        self.assertFalse(moving_threats(route, [prediction(position=(25., -3.5))], [0., 0.], 3., (2., .9)))

    def test_future_crossing_triggers_from_history(self):
        actor = prediction('pedestrian', (10., -4.), (0., 1.))
        threats = moving_threats([[0., 0.], [100., 0.]], [actor], [0., 0.], 2.5, (2., .9))
        self.assertEqual(threats[0]['kind'], 'pedestrian')

    def test_unobserved_pedestrian_motion_is_not_assumed_stationary(self):
        actor = prediction('pedestrian', (12., 3.), (0., 0.))
        path = [[0., 0.], [50., 0.]]
        result = swept_motion_check(path, [actor], 2., (2., .9), 2.)
        self.assertFalse(result['safe'])
        self.assertEqual(result['blocking_actors'][0]['kind'], 'pedestrian_motion_unobserved')
        actor.observation_count = 2
        self.assertTrue(swept_motion_check(path, [actor], 2., (2., .9), 2.)['safe'])

    def test_safe_lateral_escape_and_blocked_escape(self):
        path = [[0., 0.], [10., -3.5], [25., -3.5], [40., 0.]]
        actor = prediction(position=(30., 0.))
        safe = swept_motion_check(path, [actor], 2.5, (2., .9), 2.5)
        self.assertTrue(safe['safe'], safe)
        blocked = swept_motion_check(path, [actor, prediction(position=(10., -3.5), velocity=(0., 0.), actor_id=2)], 2.5, (2., .9), 2.5)
        self.assertFalse(blocked['safe'])
        self.assertIn(2, [item['actor_id'] for item in blocked['blocking_actors']])

    def test_unobserved_motion_allows_only_reachable_set_safe_braking(self):
        path = [[0., 0.], [50., 0.]]
        actor = prediction('pedestrian', (15., 3.), (0., 0.))
        self.assertFalse(swept_motion_check(path, [actor], 4., (2., .9), 0.)['safe'])
        self.assertTrue(swept_motion_check(path, [actor], 4., (2., .9), 0.,
                                         allow_unobserved_braking=True)['safe'])
        actor = prediction('pedestrian', (3., 2.), (0., 0.))
        self.assertFalse(swept_motion_check(path, [actor], 4., (2., .9), 0.,
                                          allow_unobserved_braking=True)['safe'])
        self.assertFalse(swept_motion_check(path, [actor], 4., (2., .9), 2.,
                                          allow_unobserved_braking=True)['safe'])

    def test_pedestrian_yield_clears_only_when_corridor_clears(self):
        path = [[0., 0.], [50., 0.]]
        self.assertTrue(pedestrian_blocks_route(path, prediction('pedestrian', (10., -3.), (0., 1.)), (2., .9)))
        self.assertFalse(pedestrian_blocks_route(path, prediction('pedestrian', (10., 3.), (0., 1.)), (2., .9)))
        self.assertFalse(pedestrian_blocks_route(path, prediction('pedestrian', (-10., 0.), (0., 1.)), (2., .9)))

    def test_braking_only_does_not_hide_oncoming_collision(self):
        actor = prediction(position=(12., 0.), velocity=(-3., 0.))
        self.assertFalse(swept_motion_check([[0., 0.], [50., 0.]], [actor], 3., (2., .9), 0.)['safe'])

    def test_diagonal_ego_axes_alone_must_not_invent_parallel_overlap(self):
        # Ego is diagonal, the other rectangle remains above its uppermost
        # corner: 3.2 - .5 - (2 + .5)/sqrt(2) = .932m true axis clearance.
        # Their intervals overlap on BOTH ego axes; the actor's axis separates.
        actor = prediction(position=(0., 3.2), velocity=(.31, 0.))
        actor.extent_xy = (2., .5)
        result = swept_motion_check([[0., 0.], [10., 10.]], [actor], 0., (2., .5), 0., margin_m=.6)
        self.assertTrue(result['safe'], result)

    def test_initial_proximity_requires_opening_and_never_accepts_body_overlap(self):
        actor = prediction(position=(0., 1.9), velocity=(0., 1.))
        actor.extent_xy = (.5, .5)
        path = [[0., 0.], [50., 0.]]
        self.assertTrue(swept_motion_check(path, [actor], 0., (2., .9), 0.,
                                         tracking_margin_m=.4)['safe'])
        stopped = prediction(position=(0., 1.9), velocity=(0., 0.))
        stopped.extent_xy = (.5, .5)
        self.assertFalse(swept_motion_check(path, [stopped], 0., (2., .9), 0.,
                                          tracking_margin_m=.4)['safe'])
        overlap = prediction(position=(0., 1.3), velocity=(0., 1.))
        overlap.extent_xy = (.5, .5)
        self.assertFalse(swept_motion_check(path, [overlap], 0., (2., .9), 0.)['safe'])

    def test_fixed_navigation_survives_queue_purge_and_detached_mutation(self):
        import carla
        planner = LocalPlanner.__new__(LocalPlanner)
        planner._waypoints_queue = deque(maxlen=100)
        planner._continuous_route_tracking = False
        planner._initial_commands = {}
        plan = [(SimpleNamespace(transform=carla.Transform(carla.Location(x=float(index * 2))),
            road_id=1, section_id=0, lane_id=-1), RoadOption.LEFT if 6 <= index <= 8 else RoadOption.LANEFOLLOW)
            for index in range(18)]
        planner.set_global_plan(plan)
        original = planner.get_initial_navigation(6., 20.)
        self.assertEqual(set(original['commands']), {0, 1})
        self.assertEqual(original['commands'][0], 0)
        self.assertEqual(original['commands'][6], 1)
        self.assertEqual(original['commands'][10], 0)
        for index in range(16):
            self.assertEqual(planner.get_navigation_command(6., 20.), original['commands'][index])
            planner._waypoints_queue.popleft()
        mutated = planner.get_initial_navigation(6., 20.)
        mutated['points'][0][0] = 999.
        mutated['commands'][0] = 99
        self.assertEqual(planner.get_initial_navigation(6., 20.), original)


if __name__ == '__main__':
    unittest.main()
