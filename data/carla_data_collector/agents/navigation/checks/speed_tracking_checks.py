"""Keep coasting optimization subordinate to actual stopping constraints."""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
from agents.navigation.speed_tracking import allow_coasting, limit_launch_throttle, waiting_coast_speed


class SpeedTrackingChecks(unittest.TestCase):
    def test_waiting_speed_retains_more_than_two_metres_of_coasting_margin(self):
        for distance in [3., 5., 15., 30.]:
            speed = waiting_coast_speed(distance)
            self.assertLess(speed * speed / (2. * .25) + 2., distance)
        self.assertEqual(waiting_coast_speed(1.), 0.)

    def test_small_launch_overshoot_coasts(self):
        self.assertTrue(allow_coasting(1.4, .7, 9., []))

    def test_nearby_stop_requires_braking(self):
        self.assertFalse(allow_coasting(2., 1.5, 0., [(3., 0.)]))

    def test_soft_goal_overshoot_coasts_but_real_obstacle_keeps_braking(self):
        self.assertFalse(allow_coasting(2.0587, 1.0552, 1.16, []))
        self.assertTrue(allow_coasting(2.0587, 1.0552, 1.16, [],
                                       soft_navigation_goal=True))
        self.assertFalse(allow_coasting(2.0587, 1.0552, 1.16, [(3., 0.)],
                                        soft_navigation_goal=True))
        self.assertFalse(allow_coasting(2., .5, .5, [], stop_for_conflict=True,
                                        soft_navigation_goal=True))

    def test_safe_admitted_maneuver_can_release_stale_wait_tracking_brake(self):
        self.assertTrue(allow_coasting(1.64, .6, 2.5, [], safe_maneuver_tracking=True))
        self.assertTrue(allow_coasting(1.64, 1.67, 2.5, [], safe_maneuver_tracking=True))
        self.assertFalse(allow_coasting(1.64, 1.67, 2.5, []))
        self.assertFalse(allow_coasting(1.64, .6, 2.5, [(2., 0.)],
                                        safe_maneuver_tracking=True))

    def test_curve_limit_above_current_speed_does_not_force_braking(self):
        self.assertTrue(allow_coasting(1.8, 1.4, 2.5, [(0., 4.2)]))

    def test_far_wait_has_room_for_coasting(self):
        self.assertTrue(allow_coasting(1.7, 1.5, 0., [], True, 20.))

    def test_unknown_conflict_cannot_coast(self):
        self.assertFalse(allow_coasting(1.7, 1.5, 0., [], True))

    def test_immediate_curve_and_large_overspeed_keep_braking(self):
        self.assertFalse(allow_coasting(2.5, 2., 1., [(0., 1.)]))
        self.assertFalse(allow_coasting(2.5, .5, .5, []))

    def test_launch_rate_limits_rise_and_releases_throttle_immediately(self):
        self.assertAlmostEqual(limit_launch_throttle(.5, .1, .1, 1.), .16)
        self.assertEqual(limit_launch_throttle(0., .5, .1, 1.), 0.)

    def test_gearbox_engagement_does_not_build_large_launch_throttle(self):
        self.assertEqual(limit_launch_throttle(.7, .7, .1, 0.), .15)
