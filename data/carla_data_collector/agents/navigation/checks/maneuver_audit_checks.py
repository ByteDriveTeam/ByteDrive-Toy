"""Do not discard the stopping tick from measured comfort statistics."""
import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
from evaluation.check_maneuver_records import audit


class ManeuverAuditChecks(unittest.TestCase):
    def test_deceleration_to_zero_is_counted(self):
        frame = {'expert': {'state': 'STOPPING', 'reason': 'follow_vehicle'},
                 'navigation_command': 0, 'raw_navigation_command': 0,
                 'recovery_command': 0, 'sim_time': .1,
                 'ego': {'transform': [0., 0., 0., 0., 0., 0.],
                         'velocity': [0., 0., 0.], 'acceleration': [-12., 0., 0.],
                         'control': {'brake': .5}},
                 'bboxes': [{'semantic': 'ego', 'id': 1, 'location': [0., 0., 0.],
                             'rotation': [0., 0., 0.], 'extent': [2., 1., .5]}]}
        record = {'scene_frames': [frame], 'trace': [{'speed_mps': 1.2}],
                  'collisions': [], 'expected_outcome': 'wait', 'frames': 1,
                  'distance_to_goal_m': 10., 'goal_reached': False}
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / 'stop.json'
            path.write_text(json.dumps(record), encoding='utf-8')
            result = audit(path)
        self.assertEqual(result['strong_deceleration_frames'], [0])
        self.assertEqual(result['max_strong_deceleration_entry_speed_mps'], 1.2)
        self.assertFalse(result['comfort_passed'])
        self.assertEqual(result['moving_hard_brake_frames'], [0])
        self.assertEqual(result['min_longitudinal_accel_mps2'], -12.)
