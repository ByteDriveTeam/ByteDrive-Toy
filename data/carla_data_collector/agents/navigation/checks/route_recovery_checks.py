"""Geometric checks for temporary bypass and route return paths."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
from agents.navigation.route_recovery import project_route, temporary_route


class RouteRecoveryChecks(unittest.TestCase):
    def setUp(self):
        self.route = [[float(x), 0.0] for x in range(0, 81, 5)]

    def test_bypass_both_sides_and_return_to_same_route(self):
        for side in (-1, 1):
            with self.subTest(side=side):
                plan = temporary_route(
                    self.route, [0.0, 0.0], side, 3.5,
                    obstacle_position=[25.0, 0.0])
                at_obstacle = min(plan["points"],
                                  key=lambda point: abs(point[0] - 25.0))
                self.assertGreater(-side * at_obstacle[1], 3.3)
                self.assertAlmostEqual(plan["points"][-1][1], 0.0,
                                       delta=0.05)
                self.assertEqual(plan["phase"], "avoid")

    def test_displaced_ego_recovers_from_either_side(self):
        for offset in (-3.5, 3.5):
            with self.subTest(offset=offset):
                plan = temporary_route(
                    self.route, [5.0, offset],
                    -1 if offset > 0 else 1, 3.5)
                self.assertEqual(plan["phase"], "recover")
                self.assertAlmostEqual(plan["ego_route_offset_m"], -offset)
                self.assertLess(abs(plan["target_world_xy"][1]),
                                abs(offset))
                self.assertAlmostEqual(plan["points"][-1][1], 0.0,
                                       delta=0.05)

    def test_projection_keeps_signed_lateral_offset(self):
        self.assertAlmostEqual(project_route(self.route, [10, 3])[1], -3.0)
        self.assertAlmostEqual(project_route(self.route, [10, -3])[1], 3.0)


if __name__ == "__main__":
    unittest.main()
