"""Synthetic forward/backward checks for parallel route and recovery inputs."""

import unittest
from unittest.mock import patch

import torch
from torch import nn

from config import load_config
from model.driving_model.unified_driving_model import _VisualProcessor, _TrajectoryDecoder


class BackboneStub(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec

    def forward(self, rgb):
        patches = (rgb.shape[-2] // self.spec.patch_size) * (rgb.shape[-1] // self.spec.patch_size)
        return rgb.new_zeros(rgb.shape[0], len(self.spec.feature_layers), patches,
                             self.spec.hidden_dim)


class ParallelRecoveryChecks(unittest.TestCase):
    def test_both_inputs_reach_visual_and_trajectory_with_gradients(self):
        torch.manual_seed(17)
        cfg = load_config("config/default.yaml")
        spec = cfg.model.unified_driving
        spec.dim, spec.ffn_hidden, spec.condition_hidden, spec.geometry_hidden = 96, 128, 64, 32
        spec.self_heads_3d, spec.cross_heads = 2, 2
        spec.action_vocab_size, spec.action_attention_rank = 8, 4
        bb = cfg.model.dinov3_backbone
        bb.hidden_dim, bb.patch_size = 16, 8
        with patch("model.driving_model.unified_driving_model.DinoV3Backbone", BackboneStub):
            visual = _VisualProcessor(cfg)
        artifact = {"actions": torch.randn(8, 30, 2),
                    "action_mean": torch.zeros(30, 2), "action_std": torch.ones(30, 2),
                    "residual_mean": torch.zeros(30, 2), "residual_std": torch.ones(30, 2),
                    "motion_mean": torch.zeros(4), "motion_std": torch.ones(4)}
        decoder = _TrajectoryDecoder(spec, artifact)
        rgb = torch.zeros(2, 4, 3, 32, 32)
        navigation = torch.ones(2, dtype=torch.long)  # same original left-turn command
        recovery = torch.tensor([1, 2])
        ego = torch.zeros(2, 4)
        intrinsics = torch.tensor([20., 20., 16., 16.]).expand(2, 4, 4)
        motion = torch.eye(4).expand(2, 4, 4, 4)
        tokens, _, _, annotation = visual(rgb, navigation, ego, intrinsics, motion, recovery)
        result = decoder(tokens, navigation, ego, recovery_command=recovery)
        self.assertFalse(torch.allclose(tokens[0], tokens[1]))
        self.assertFalse(torch.allclose(result["condition"][0], result["condition"][1]))
        loss = result["logits"].square().mean() + annotation["state_logits"].square().mean()
        loss.backward()
        for embedding in (visual.recovery_token, decoder.recovery):
            self.assertGreater(float(embedding.weight.grad[1:].abs().sum()), 0.)
            self.assertEqual(float(embedding.weight.grad[0].abs().sum()), 0.)
        self.assertEqual(visual.navigation_token.num_embeddings, 5)
        self.assertEqual(decoder.navigation.num_embeddings, 5)
        zero = torch.zeros_like(recovery)
        with torch.no_grad():
            absent = visual(rgb, navigation, ego, intrinsics, motion)[0]
            explicit = visual(rgb, navigation, ego, intrinsics, motion, zero)[0]
        torch.testing.assert_close(absent, explicit)


if __name__ == "__main__":
    unittest.main()
