"""Static props and environment solids must survive collision supervision."""
import unittest
import numpy as np
from config import load_config
from data.driving_cache.trajectory_risk import SceneTrajectoryRisk


def bbox(x, z=1., semantic='static', source='actor'):
    return {'id': 9, 'semantic': semantic, 'source': source,
            'location': [x, 0., z], 'extent': [.5, .5, .5], 'rotation': [0., 0., 0.]}


class StaticRiskChecks(unittest.TestCase):
    def test_prop_environment_collision_and_overhead_exclusion(self):
        evaluator = SceneTrajectoryRisk.__new__(SceneTrajectoryRisk)
        evaluator.cfg = load_config('config/default.yaml')
        evaluator.divider_index = evaluator.boundary_index = None
        evaluator._offroad = lambda corners, width: np.zeros(len(corners), dtype=bool)
        ego = bbox(0., semantic='ego')
        frame = {'ego': {'transform': [0., 0., 0., 0., 0., 0.]}, 'bboxes': [ego]}
        local = np.zeros((1, 30, 2)); local[0, :, 0] = np.arange(1, 31) * .2
        headings, actions = np.zeros((1, 30)), np.zeros((1, 30, 2))
        for in_meta, height, expected in [(False, 1., True), (True, 1., True),
                                           (False, 8., False), (True, 8., False)]:
            with self.subTest(in_meta=in_meta, height=height):
                obstacle = bbox(3., z=height, source='env' if in_meta else 'actor')
                evaluator.meta = {'static_bboxes': [obstacle] if in_meta else []}
                for name in ['_solid_static_boxes', '_solid_static_bounds']:
                    if hasattr(evaluator, name): delattr(evaluator, name)
                future = [{'bboxes': [ego] + ([] if in_meta else [obstacle])} for _ in range(30)]
                flags = evaluator._risk_flags(local, headings, actions, frame, future, np.zeros(2))
                self.assertEqual(bool(flags[0, 0]), expected)
