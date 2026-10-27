"""Lane direction, boundary and visibility admission regression checks."""

import pathlib
import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
import carla
from agents.navigation.maneuver_lanes import (adjacent_candidates, check_opposing_sight,
                                            check_path_geometry, point_in_driving_corridor)
from agents.navigation.route_recovery import temporary_route


def lane(lane_id=-1, y=0., yaw=0., marking=None):
    mark = marking or NS(type=carla.LaneMarkingType.Broken, lane_change=carla.LaneChange.Both)
    return NS(road_id=1, lane_id=lane_id, lane_type=carla.LaneType.Driving,
              lane_width=3.5, is_junction=False,
              transform=carla.Transform(carla.Location(y=y), carla.Rotation(yaw=yaw)),
              left_lane_marking=mark, right_lane_marking=mark,
              get_left_lane=lambda: None, get_right_lane=lambda: None)


class ManeuverLaneChecks(unittest.TestCase):
    def test_shared_border_uses_union_without_extending_outer_road_edge(self):
        origin, adjacent = lane(-1, 0.), lane(-2, 3.5)
        adjacent.get_left_lane = lambda: origin
        origin.get_right_lane = lambda: adjacent
        road_map = NS(get_waypoint=lambda *args, **kwargs: adjacent)
        self.assertTrue(point_in_driving_corridor(road_map, carla.Location(y=1.72)))
        self.assertFalse(point_in_driving_corridor(road_map, carla.Location(y=5.32)))

    def test_mixed_directions_and_broken_opposing_border(self):
        origin = lane()
        opposing = lane(1, -3.5, 180.)
        origin.get_left_lane = lambda: opposing
        origin.get_right_lane = lambda: lane(-2, 3.5)
        self.assertEqual(len(adjacent_candidates(origin)), 1)
        choices = adjacent_candidates(origin, True)
        self.assertEqual(len(choices), 2)
        self.assertEqual([x["same_direction"] for x in choices], [False, True])
        origin.left_lane_marking = NS(type=carla.LaneMarkingType.SolidSolid,
                                     lane_change=carla.LaneChange.NONE)
        self.assertEqual(len(adjacent_candidates(origin, True)), 1)

    def test_opposing_sight_rejects_occluder_and_junction(self):
        receiving = lane(1, -3.5, 180.)
        receiving.previous = lambda distance: [lane(1, -3.5, 180.)]
        world = NS(cast_ray=Mock(return_value=[NS(location=carla.Location(x=3., y=-3.5, z=1.2))]))
        self.assertFalse(check_opposing_sight(world, receiving, 100., carla.Transform(), NS(x=2.4, y=1.))['safe'])
        junction = lane(1, -3.5, 180.)
        junction.is_junction = True
        receiving.previous = lambda distance: [junction]
        self.assertFalse(check_opposing_sight(world, receiving, 100., carla.Transform(), NS(x=2.4, y=1.))['safe'])

    def test_bypass_requires_permission_to_return(self):
        origin, receiving = lane(), lane(-2, -3.5)
        origin.get_left_lane = lambda: receiving
        receiving.right_lane_marking = NS(type=carla.LaneMarkingType.Solid,
                                          lane_change=carla.LaneChange.NONE)
        self.assertEqual(adjacent_candidates(origin, True), [])

    def test_sight_origin_stays_at_ego_not_borrowed_lane(self):
        receiving = lane(1, -3.5, 180.)
        receiving.previous = lambda distance: [lane(1, -3.5, 180.)]
        world = NS(cast_ray=Mock(return_value=[]))
        check_opposing_sight(world, receiving, 100., carla.Transform(), NS(x=2.4, y=1.))
        source, _ = world.cast_ray.call_args_list[0][0]
        self.assertLess(abs(source.y), 1.)
        self.assertLess(source.x, 3.)

    def test_incomplete_bypass_return_rejected(self):
        points = [[0., 0.], [30., 0.]]
        path = temporary_route(points, [0., 0.], 1, 3.5, obstacle_position=[25., 0.])
        self.assertFalse(check_path_geometry(None, points, path, lane(), lane(1),
                                            NS(x=2.4, y=1.))['safe'])


if __name__ == '__main__':
    unittest.main()
