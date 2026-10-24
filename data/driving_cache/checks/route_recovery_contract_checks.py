"""Decode checks for expanded route commands and expert classes."""

import unittest

import torch

from config import load_config
from data.driving_cache.driving_cache import DrivingSliceCache
from model.driving_model.checkpoint_compat import compatible_driving_state


class RouteRecoveryContractChecks(unittest.TestCase):
    def test_new_commands_and_labels_store_and_decode(self):
        cfg = load_config("config/default.yaml")
        cfg.data.driving_cache.dir = "output/stage3-cache-contract"
        cfg.data.driving_cache.version = "stage3-contract-check"
        cache = DrivingSliceCache(cfg)
        self.assertIsNone(cache.load("missing-review-sample"))
        self.assertNotEqual(cache.namespace, cache._namespace("v14-route-recovery", cfg))
        words = torch.zeros(cfg.model.unified_driving.action_vocab_size, 30, 2)
        sample = {
            "rgb": torch.zeros(4, 1, 3, 4, 4),
            "intrinsics": torch.eye(3)[None],
            "ego_state": torch.zeros(4),
            "motion_matrices": torch.eye(4)[None].repeat(4, 1, 1),
            "trajectory": torch.zeros(30, 2),
            "bev_grid": torch.zeros(1, 2, 2),
            "timestamp": 0.0, "scene": "stage3-synthetic", "index": 0,
            "word_risks": torch.zeros(len(words), dtype=torch.bool),
            "gt_risks": torch.zeros(30, dtype=torch.bool),
        }
        for command, state, reason in ((1, 10, 17), (2, 9, 16)):
            with self.subTest(command=command):
                sample["navigation_command"] = 1  # left turn can coexist with either recovery
                sample["recovery_command"] = command
                sample["frame_meta"] = {"expert": {
                    "state_id": state, "reason_code": reason,
                    "risk": {"predicted_conflict_probability": 0.2},
                    "lane_change": {"available_left": True,
                                    "available_right": True},
                }}
                key = "recovery-{}".format(command)
                cache.store(key, sample, words, torch.zeros(30, 2),
                            torch.ones(30, 2))
                decoded = cache.load(key)
                self.assertIsNotNone(decoded)
                self.assertEqual(decoded["navigation_command"].item(), 1)
                self.assertEqual(decoded["recovery_command"].item(), command)
                self.assertEqual(decoded["annotation_state"].item(), state)
                self.assertEqual(decoded["annotation_reason"].item(), reason)
                self.assertEqual(tuple(decoded["annotation_probabilities"].shape),
                                 (3,))
                sample["navigation_command"] = command + 4
                sample["raw_navigation_command"] = 2
                cache.store("legacy-{}".format(command), sample, words,
                            torch.zeros(30, 2), torch.ones(30, 2))
                legacy = cache.load("legacy-{}".format(command))
                self.assertEqual(int(legacy["navigation_command"]), 2)
                self.assertEqual(int(legacy["recovery_command"]), command)

    def test_checkpoint_expansion_keeps_existing_rows(self):
        current = {
            "visual.navigation_token.weight": torch.zeros(5, 4),
            "trajectory_decoder.navigation.weight": torch.zeros(5, 4),
            "visual.annotation_state_head.weight": torch.zeros(11, 4),
            "visual.annotation_reason_head.bias": torch.zeros(18),
        }
        old = {
            "visual.navigation_token.weight": torch.ones(5, 4),
            "trajectory_decoder.navigation.weight": torch.ones(5, 4) * 2,
            "visual.annotation_state_head.weight": torch.ones(7, 4) * 3,
            "visual.annotation_reason_head.bias": torch.ones(14) * 4,
        }
        compatible, exact_names = compatible_driving_state(current, old)
        self.assertEqual(exact_names, {"visual.navigation_token.weight",
                                      "trajectory_decoder.navigation.weight"})
        for name, value in old.items():
            torch.testing.assert_close(compatible[name][:len(value)], value)
            self.assertEqual(tuple(compatible[name].shape),
                             tuple(current[name].shape))


if __name__ == "__main__":
    unittest.main()
