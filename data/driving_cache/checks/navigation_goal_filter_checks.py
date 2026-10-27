"""Terminal selection, vocabulary boundaries and real cache decode checks."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from config import load_config
from data.driving_cache.driving_cache import DrivingSliceCache, DrivingSliceDataset, prepare_driving_cache
from data.navigation_goal_filter import (navigation_goal_cutoff,
    navigation_training_prefix, window_before_navigation_goal)
from data.trajectory_vocabulary.trajectory_vocabulary import _action_windows


def timeline():
    return [{"sim_time": i / 10., "ego": {
        "transform": [i / 10., 0., 0., 0., 0., 0.],
        "velocity": [1., 0., 0.], "acceleration": [0., 0., 0.]}, "expert": {}}
        for i in range(101)]


def mark_goal(row):
    row["expert"] = {"longitudinal_plan": {"navigation_goal_active": True}}


class NavigationGoalFilterChecks(unittest.TestCase):
    def test_future_boundary_and_first_goal_decision(self):
        rows = timeline()
        mark_goal(rows[80])
        rows[80]["expert"]["decision_sim_time"] = 7.9
        self.assertEqual(navigation_goal_cutoff(rows), 7.9)
        self.assertTrue(window_before_navigation_goal(4.8, 3., 7.9))
        self.assertFalse(window_before_navigation_goal(4.9, 3., 7.9))
        self.assertFalse(window_before_navigation_goal(9., 3., 7.9))

    def test_light_stops_legacy_rows_and_old_goal_records(self):
        rows = timeline()
        rows[40]["expert"] = {"state": "APPROACH_STOP", "reason": "red_light",
            "longitudinal_plan": {"navigation_goal_distance_m": 15.,
                                   "navigation_goal_speed_mps": 4.}}
        self.assertIsNone(navigation_goal_cutoff(rows))
        self.assertIs(navigation_training_prefix(rows), rows)
        rows[40]["expert"]["reason"] = "comfortable_braking"
        self.assertEqual(navigation_goal_cutoff(rows), 4.)
        rows[40]['expert']['longitudinal_plan']['navigation_goal_active'] = False
        self.assertIsNone(navigation_goal_cutoff(rows))  # A static stop is not a destination stop.

    def test_low_rate_fallback_and_vocabulary_truncate_before_interpolation(self):
        rows = timeline()
        frames = [{"sim_time": 8., "expert": {}}]
        mark_goal(frames[0])
        prefix = navigation_training_prefix(rows, frames)
        self.assertLess(prefix[-1]["sim_time"], 8.)
        cfg = load_config("config/default.yaml").trajectory_vocabulary
        explicit = timeline()
        mark_goal(explicit[80])
        expected = _action_windows(prefix, cfg)
        actual = _action_windows(explicit, cfg)
        self.assertIsNotNone(actual)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)
        self.assertLess(len(actual[0]), len(_action_windows(rows, cfg)[0]))

    def test_prepare_rejects_future_before_risk_and_rgb_work(self):
        cfg = load_config("config/default.yaml")
        rows = timeline()
        mark_goal(rows[80])
        class Reader:
            def __init__(self, *args):
                self.reader = type("Raw", (), {"meta": {
                    "complete": True, "failed": False, "status": "ok"}})()
                self.frames, self.kinematics = rows, rows
            def close(self):
                pass
            def gt_trajectory(self, *args):
                raise AssertionError("A window crossing the goal must be skipped first")
        with tempfile.TemporaryDirectory(dir="output") as temp:
            cfg.data.driving_cache.dir = str(Path(temp).resolve() / "cache")
            cfg.trajectory_vocabulary.output_dir = str(Path(temp).resolve())
            cfg.trajectory_vocabulary.action_output = "words.npz"
            np.savez(Path(temp) / "words.npz", actions=np.zeros((2, 30, 2)),
                     residual_mean=np.zeros((30, 2)), residual_std=np.ones((30, 2)))
            with patch("data.driving_cache.driving_cache.SceneTrajectoryRisk", Reader):
                stats = prepare_driving_cache([
                    {"scene": temp, "index": 50, "timestamp": 5.}], cfg)
            self.assertEqual(stats["samples"], 0)
            self.assertEqual(stats["skipped_navigation_goal"], 1)
            self.assertEqual(stats["skipped_missing_risk_gt"], 0)

    def test_selection_namespace_preserves_old_files_and_real_decode(self):
        cfg = load_config("config/default.yaml")
        with tempfile.TemporaryDirectory(dir="output") as temp:
            cfg.data.driving_cache.dir = str(Path(temp).resolve())
            with patch("data.driving_cache.driving_cache.TRAINING_SELECTION", "old-policy"):
                old = DrivingSliceCache(cfg)
            old_file = old._file("keep")
            torch.save({"old_payload": torch.ones(1)}, old_file)
            cache = DrivingSliceCache(cfg)
            self.assertNotEqual(cache.path, old.path)
            self.assertIsNone(cache.load("keep"))
            words = torch.zeros(cfg.model.unified_driving.action_vocab_size, 30, 2)
            sample = {"rgb": torch.zeros(4, 1, 3, 4, 4), "intrinsics": torch.eye(3)[None],
                "ego_state": torch.zeros(4), "motion_matrices": torch.eye(4)[None].repeat(4, 1, 1),
                "trajectory": torch.zeros(30, 2), "bev_grid": torch.zeros(1, 2, 2),
                "timestamp": 0., "scene": "synthetic", "index": 0,
                "word_risks": torch.zeros(len(words)), "gt_risks": torch.zeros(30),
                "navigation_command": 1, "recovery_command": 2}
            cache.store("keep", sample, words, torch.zeros(30, 2), torch.ones(30, 2))
            decoded = cache.load("keep")
            self.assertEqual(int(decoded["navigation_command"]), 1)
            self.assertEqual(int(decoded["recovery_command"]), 2)
            self.assertEqual(tuple(decoded["trajectory_residual"].shape), (30, 2))
            dataset = DrivingSliceDataset(cfg)
            self.assertEqual(len(dataset), 1)
            torch.testing.assert_close(dataset[0]["rgb"], sample["rgb"])
            self.assertTrue(old_file.exists())
            self.assertEqual(old.load("keep")["old_payload"].item(), 1.)
            sample["expert"] = {"longitudinal_plan": {"navigation_goal_active": True}}
            with self.assertRaisesRegex(ValueError, "goal approach"):
                cache.store("invalid", sample, words, torch.zeros(30, 2), torch.ones(30, 2))
            self.assertIsNone(cache.load("invalid"))


if __name__ == "__main__":
    unittest.main()
