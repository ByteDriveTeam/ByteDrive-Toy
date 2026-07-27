import unittest
from types import SimpleNamespace

import torch

from data.lidar_voxelization import lidar_xyz_to_voxels


class LidarVoxelizationTest(unittest.TestCase):
    def setUp(self):
        self.bev = SimpleNamespace(
            x_min_m=-1.0,
            x_max_m=1.0,
            y_min_m=-1.0,
            y_max_m=1.0,
            z_min_m=-1.0,
            z_max_m=1.0,
        )
        self.ego_box = {
            "transform": [
                [1.0, 0.0, 0.0, 100.0],
                [0.0, 1.0, 0.0, 100.0],
                [0.0, 0.0, 1.0, 100.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "extent": [0.1, 0.1, 0.1],
        }

    def voxelize(self, points):
        return lidar_xyz_to_voxels(
            points,
            lidar_extrinsic=[0.0, 0.0, 0.0],
            ego_box=self.ego_box,
            bev=self.bev,
            voxel_size_m=0.5,
        )

    def test_uses_each_voxel_center_for_local_statistics(self):
        points = torch.tensor([
            [0.1, -0.9, 0.6],
            [0.4, -0.6, 0.9],
            [-0.7, 0.6, -0.1],
        ])

        stats, occupied = self.voxelize(points)

        self.assertEqual(tuple(stats.shape), (6, 4, 4, 4))
        self.assertEqual(tuple(occupied.shape), (1, 4, 4, 4))
        self.assertEqual(int(occupied.sum()), 2)
        torch.testing.assert_close(
            stats[:, 3, 1, 0],
            torch.tensor([0.0, 0.0, 0.0, 0.15, 0.15, 0.15]),
            atol=1e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            stats[:, 1, 3, 3],
            torch.tensor([0.05, -0.15, 0.15, 0.0, 0.0, 0.0]),
            atol=1e-6,
            rtol=0.0,
        )

    def test_empty_cloud_keeps_zero_stats_and_unoccupied_mask(self):
        stats, occupied = self.voxelize(torch.empty((0, 3)))

        self.assertEqual(tuple(stats.shape), (6, 4, 4, 4))
        self.assertEqual(tuple(occupied.shape), (1, 4, 4, 4))
        self.assertFalse(bool(stats.any()))
        self.assertFalse(bool(occupied.any()))


if __name__ == "__main__":
    unittest.main()
