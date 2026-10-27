"""Static footprints, return safety and height-aware native/prop queries."""
import pathlib
import sys
import unittest
from types import SimpleNamespace

import carla
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
from agents.navigation.static_obstacles import StaticObstacleIndex, actor_box
from agents.navigation.maneuver_safety import assess_maneuver, choose_candidate
from agents.navigation.risk_field import plan_front_vehicle_speed
from common.obstacle_geometry import path_hits_box


def box(x, y, z=1., yaw=0., extent=(.5, .2, .5)):
    value = carla.BoundingBox(carla.Location(x=x, y=y, z=z), carla.Vector3D(*extent))
    value.rotation.yaw = yaw
    return value


class StaticObstacleChecks(unittest.TestCase):
    def test_finite_sweep_checks_rotated_obstacle_and_complete_return(self):
        payload = {'location': [15., 0., 1.], 'extent': [.2, 2., .5], 'rotation': [0., 0., 0.]}
        self.assertTrue(path_hits_box([[0., 0.], [30., 0.]], payload, (2., 1.)))
        self.assertFalse(path_hits_box([[0., -4.], [30., -4.]], payload, (2., 1.)))
        self.assertTrue(path_hits_box([[0., -4.], [12., -4.], [18., 0.], [30., 0.]],
                                      payload, (2., 1.)))
        payload['location'] = [-15., 0., 1.]
        self.assertFalse(path_hits_box([[0., -4.], [30., 0.]], payload, (2., 1.)))

    def test_height_filter_static_predictions_and_stable_native_id(self):
        native = box(10., 0.)
        overhead = box(15., 0., z=8.)
        flat = box(5., 0., extent=(1., 1., .001))
        world = SimpleNamespace(get_level_bbs=lambda label: [native, overhead, flat]
                                 if label == carla.CityObjectLabel.Static else [],
                                get_actors=lambda: SimpleNamespace(filter=lambda _: []))
        ego = SimpleNamespace(id=1, bounding_box=box(0., 0., z=1., extent=(2., 1., .7)),
                              get_transform=lambda: carla.Transform())
        index = StaticObstacleIndex(world)
        first = index.predictions(ego)
        self.assertEqual(len(first), 1)
        self.assertLess(first[0].actor_id, 0)
        self.assertEqual(first[0].actor_id, index.predictions(ego)[0].actor_id)
        self.assertEqual(first[0].actor_type, 'static')
        self.assertEqual(first[0].modes, ())
        np.testing.assert_array_equal(first[0].covariances, 0.)
        result = assess_maneuver([[0., 0.], [40., 0.]], [[0., 0.], [40., 0.]],
            first, [0., 0.], 2., (2., 1.), 0.)
        self.assertFalse(result['safe'])
        self.assertEqual(result['blocking_actors'][0]['kind'], 'static_on_path')

    def test_static_prop_local_box_offset_is_transformed(self):
        actor = SimpleNamespace(id=5, bounding_box=box(1., 0., z=.5, yaw=15.),
            get_transform=lambda: carla.Transform(carla.Location(x=10., y=20.), carla.Rotation(yaw=90.)))
        payload = actor_box(actor)
        self.assertAlmostEqual(payload['location'][0], 10., places=5)
        self.assertAlmostEqual(payload['location'][1], 21., places=5)
        self.assertEqual(payload['rotation'][2], 105.)

    def test_safe_in_lane_candidate_is_preferred_even_during_confirmation(self):
        short = {'safe': True, 'ready': False, 'side': 1, 'in_lane': True, 'cost': 1.}
        borrow = {'safe': True, 'ready': True, 'side': -1, 'cost': .1}
        self.assertIsNone(choose_candidate([short, borrow]))
        short['ready'] = True
        self.assertIs(choose_candidate([short, borrow]), short)

    def test_static_front_stop_does_not_block_a_clear_finite_bypass(self):
        world = SimpleNamespace(get_level_bbs=lambda label: [box(20., 0.)]
                                 if label == carla.CityObjectLabel.Static else [],
                                get_actors=lambda: SimpleNamespace(filter=lambda _: []))
        ego = SimpleNamespace(id=1, bounding_box=box(0., 0., z=1., extent=(2., 1., .7)),
                              get_transform=lambda: carla.Transform())
        predictions = StaticObstacleIndex(world).predictions(ego)
        def front(path):
            return plan_front_vehicle_speed(path, predictions, [0., 0.], 2., 2., 1., .35, .7, 3.)
        self.assertEqual(front([[0., 0.], [40., 0.]])['kind'], 'static')
        self.assertIsNone(front([[0., 0.], [8., -4.], [26., -4.], [35., 0.]])['kind'])
