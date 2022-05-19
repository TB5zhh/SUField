"""
Voxelization

Pre-voxelize steps:
1. Clipping in a box (TODO cause augmented views have different points)
2. Random rotation
3. Random scaling
4. Translation to non-negative region

Voxelize:
ME.utils.sparse_quantize

IMPORTANT: Voxelization does NOT change point order or removes points
"""

import MinkowskiEngine as ME
import numpy as np
import torch
from scipy.linalg import expm, norm

base_vec = lambda x: np.array((1, 0, 0)) if x == 0 else (np.array((0, 1, 0)) if x == 1 else np.array((0, 0, 1)))


def rotate_mat(axis, theta):
    """
    Return rotation matrix alone certain `axis` by `angle`
    """
    return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

class Voxelizer():

    # ScanNet: z axis is vertical
    ROTATION_BOUNDARIES = [
        (-np.pi / 64, np.pi / 64),
        (-np.pi / 64, np.pi / 64),
        (-np.pi, np.pi),
    ]
    SCALE_BOUNDARIES = (0.9, 1.1)
    VOXEL_SIZE = 0.05
    def __init__(self, fix_map=None) -> None:
        self.fix_map = fix_map

    def get_random_rotation_mat(self):
        rotation_mat = torch.eye(3)
        for axis_index, boundary in enumerate(self.ROTATION_BOUNDARIES):
            theta = np.random.uniform(*boundary)
            axis = base_vec(axis_index)
            rotation_mat = rotation_mat @ rotate_mat(axis, theta)
        homo_mat = torch.eye(4)
        homo_mat[:3, :3] = rotation_mat
        return homo_mat

    def get_random_scaling_mat(self):
        homo_mat = torch.eye(4)
        scale = np.random.uniform(*self.SCALE_BOUNDARIES)
        homo_mat[:3, :3].fill_diagonal_(scale)
        return homo_mat

    def get_translation_mat(self, coords):
        min_coords = coords[:, :3].min(0)[0]
        homo_mat = torch.eye(4)
        homo_mat[:3, -1] = -min_coords
        return homo_mat

    def __call__(self, coords, feats, labels, *args):
        point_count, _ = coords.shape
        homo_coords = torch.hstack((coords, torch.ones((point_count, 1), dtype=coords.dtype)))
        rotate_mat = self.get_random_rotation_mat()
        scale_mat = self.get_random_scaling_mat()
        homo_coords = homo_coords @ (rotate_mat @ scale_mat).T
        trans_mat = self.get_translation_mat(homo_coords)
        transformed_coords = torch.floor(homo_coords @ (trans_mat).T[:, :3] / self.VOXEL_SIZE)
        if self.fix_map is None:
            voxelized_coords, voxelized_feats, voxelized_labels, indices_map, reverse_indices_map = ME.utils.sparse_quantize(
                transformed_coords, feats, labels, return_index=True, return_inverse=True
            )
        else: 
            voxelized_coords, voxelized_feats, voxelized_labels, indices_map, reverse_indices_map = \
                transformed_coords[self.fix_map], feats[self.fix_map], labels[self.fix_map], self.fix_map, None
        return voxelized_coords, voxelized_feats, voxelized_labels, *args[1:], indices_map, reverse_indices_map
