"""Continuous tracking must preserve route order through intersections."""

import pathlib
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
from agents.navigation.route_tracking import RouteTracker
from common.navigation_command import navigation_command_from_plan


class RouteTrackingChecks(unittest.TestCase):
    def test_lateral_recovery_does_not_change_straight_route_command(self):
        plan = []
        for x in (0., 10., 20., 30.):
            transform = SimpleNamespace(location=SimpleNamespace(x=x, y=0.),
                get_forward_vector=lambda: SimpleNamespace(x=1., y=0.))
            plan.append((SimpleNamespace(transform=transform, road_id=1,
                                         section_id=0, lane_id=-1), 4))
        for y in (-3.5, 0., 3.5):
            self.assertEqual(navigation_command_from_plan(
                SimpleNamespace(x=-2., y=y), plan, 25., 20.), 0)

    def test_target_moves_continuously_across_sparse_waypoints(self):
        tracker = RouteTracker([[0, 0], [5, 0], [10, 0], [15, 0]])
        before = tracker.target([4.99, 0], 2.)
        after = tracker.target([5.01, 0], 2.)
        self.assertAlmostEqual(before[0], 6.99)
        self.assertAlmostEqual(after[0] - before[0], .02)

    def test_crossing_does_not_jump_to_distant_future_segment(self):
        points = [[float(x), 0.] for x in range(30)]
        points += [[29., float(y)] for y in range(1, 20)]
        points += [[float(x), 19.] for x in range(28, -1, -1)]
        points += [[0., float(y)] for y in range(18, -2, -1)]
        tracker = RouteTracker(points)
        tracker.target([0., .01], 2.)
        self.assertLess(tracker.index, 3)
        tracker.target([1., 0.], 2.)
        arc = tracker.arc
        tracker.target([.8, 0.], 2.)
        self.assertGreaterEqual(tracker.arc, arc)


if __name__ == "__main__":
    unittest.main()
