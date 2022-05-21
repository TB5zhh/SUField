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
from scipy.linalg import expm, norm

class Voxelizer():

    def __init__(self, fix_map=None, voxel_size=0.05) -> None:
        self.fix_map = fix_map
        self.voxel_size = voxel_size

    def __call__(self, coords, feats, labels, *args):
        # point_count, _ = coords.shape
        # homo_coords = np.hstack((coords, np.ones((point_count, 1), dtype=coords.dtype)))
        # rotate_mat = self.get_random_rotation_mat()
        # scale_mat = self.get_random_scaling_mat()
        # homo_coords = homo_coords @ (rotate_mat @ scale_mat).T
        # trans_mat = self.get_translation_mat(homo_coords)
        # transformed_coords = np.floor(homo_coords @ (trans_mat).T[:, :3] / self.VOXEL_SIZE)
        transformed_coords = coords / self.voxel_size
        if self.fix_map is None:
            voxelized_coords, voxelized_feats, voxelized_labels, indices_map, reverse_indices_map = ME.utils.sparse_quantize(
                transformed_coords, feats, labels, return_index=True, return_inverse=True
            )
        else: 
            voxelized_coords, voxelized_feats, voxelized_labels, indices_map, reverse_indices_map = \
                transformed_coords[self.fix_map], feats[self.fix_map], labels[self.fix_map], self.fix_map, None

        return np.asarray(voxelized_coords), np.asarray(voxelized_feats), np.asarray(voxelized_labels), *args, np.asarray(indices_map), np.asarray(reverse_indices_map),
