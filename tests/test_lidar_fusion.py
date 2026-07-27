import unittest
from types import SimpleNamespace

import torch

from model.lidar_fusion import LidarQueryFusion


def _config():
    return SimpleNamespace(
        work_dim=2,
        bev=SimpleNamespace(
            x_min_m=0.0,
            x_max_m=1.0,
            y_min_m=-0.5,
            y_max_m=0.5,
            z_min_m=0.0,
            z_max_m=0.5,
            height=1,
            width=1,
        ),
        lidar_fusion=SimpleNamespace(
            voxel_size_m=0.5,
            voxel_embed_dim=2,
            height_hidden_dim=2,
            reduced_dim=2,
            gate_hidden_dim=2,
        ),
    )


class LidarFusionTest(unittest.TestCase):
    def test_scales_linearly_without_clamping_and_preserves_bypasses(self):
        fusion = LidarQueryFusion(_config())
        with torch.no_grad():
            fusion.voxel_projection.weight.fill_(1.0)
            fusion.voxel_projection.bias.zero_()
            fusion.height_reducer[0].weight.fill_(1.0)
            fusion.height_reducer[0].bias.zero_()
            fusion.height_reducer[2].weight.fill_(1.0)
            fusion.height_reducer[2].bias.zero_()
            fusion.spatial_alignment.weight.fill_(1.0)
            fusion.spatial_alignment.bias.zero_()
            fusion.gate[-1].weight.zero_()
            fusion.gate[-1].bias.zero_()

        query = torch.zeros((2, 2, 1, 1))
        visual = torch.zeros((2, 1, 2, 1, 1))
        stats = torch.full((2, 6, 1, 2, 2), 0.5)
        occupied = torch.ones((2, 1, 1, 2, 2), dtype=torch.bool)
        valid = torch.tensor([1.0, 0.0])
        projection_inputs = []

        handle = fusion.voxel_projection.register_forward_pre_hook(
            lambda _module, inputs: projection_inputs.append(inputs[0].detach().clone())
        )
        try:
            output = fusion(query, visual, stats, occupied, valid)
            self.assertEqual(len(projection_inputs), 1)
            torch.testing.assert_close(
                projection_inputs[0],
                stats * 4.0,
                atol=0.0,
                rtol=0.0,
            )
            self.assertTrue(bool((projection_inputs[0] > 1.0).all()))
            self.assertTrue(bool((output[0] != query[0]).any()))
            torch.testing.assert_close(output[1], query[1], atol=0.0, rtol=0.0)

            all_invalid = torch.zeros_like(valid)
            bypassed = fusion(query, visual, stats, occupied, all_invalid)
            self.assertIs(bypassed, query)
            self.assertEqual(len(projection_inputs), 1)
            self.assertIs(fusion(query, visual), query)
        finally:
            handle.remove()


if __name__ == "__main__":
    unittest.main()
