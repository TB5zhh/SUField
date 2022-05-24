import random

import logging
import numpy as np
import scipy
import scipy.ndimage
import scipy.interpolate
import torch
import math
import torchvision.transforms as transforms

import MinkowskiEngine as ME
from scipy.linalg import expm, norm

# A sparse tensor consists of coordinates and associated features.
# You must apply augmentation to both.
# In 2D, flip, shear, scale, and rotation of images are coordinate transformation
# color jitter, hue, etc., are feature transformations
##############################
# Feature transformations
##############################
from IPython import embed

from sufield.lib.utils.distributed import get_rank

class AbstractTransform:
    COORD_DIM = 3
    TRANSFORM = 392
    def __init__(self) -> None:
        ...
        
    def __call__(self, coords, feats, labels):
        raise NotImplementedError


class ToTensor(AbstractTransform):
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, coords, feats, labels):
        return torch.as_tensor(coords), torch.as_tensor(feats), torch.as_tensor(labels)
    


class ChromaticTranslation(AbstractTransform):
    """Add random color to the image, input must be an array in [0,255] or a PIL image"""

    def __init__(self, trans_range_ratio=1e-1, apply_ratio=0.95):
        """
        trans_range_ratio: ratio of translation i.e. 255 * 2 * ratio * rand(-0.5, 0.5)
        """
        self.trans_range_ratio = trans_range_ratio
        self.apply_ratio = apply_ratio

    def __call__(self, coords, feats, labels):
        if random.random() < self.apply_ratio:
            tr = (torch.rand(1, 3, device=feats.device) - 0.5) * \
                255 * 2 * self.trans_range_ratio
            feats[:, :3] = torch.clip(tr + feats[:, :3], 0, 255)
        return coords, feats, labels


class ChromaticAutoContrast(AbstractTransform):

    def __init__(self, randomize_blend_factor=True, blend_factor=0.5, apply_ratio=0.2):
        self.randomize_blend_factor = randomize_blend_factor
        self.blend_factor = blend_factor
        self.apply_ratio = apply_ratio

    def __call__(self, coords, feats, labels):
        if random.random() < self.apply_ratio:
            # mean = np.mean(feats, 0, keepdims=True)
            # std = np.std(feats, 0, keepdims=True)
            # lo = mean - std
            # hi = mean + std
            lo = feats[:, :3].min(0, keepdims=True)[0]
            hi = feats[:, :3].max(0, keepdims=True)[0]
            assert hi.max() > 1, f"invalid color value. Color is supposed to be [0-255]"

            scale = 255 / (hi - lo)

            contrast_feats = (feats[:, :3] - lo) * scale

            blend_factor = random.random() if self.randomize_blend_factor else self.blend_factor
            feats[:, :3] = (1 - blend_factor) * feats + \
                blend_factor * contrast_feats
        return coords, feats, labels


class ChromaticJitter(AbstractTransform):

    def __init__(self, std=0.01, apply_ratio=0.95):
        self.std = std
        self.apply_ratio = apply_ratio

    def __call__(self, coords, feats, labels):
        if random.random() < self.apply_ratio:
            noise = torch.randn(feats.shape[0], 3, device=feats.device)
            noise *= self.std * 255
            feats[:, :3] = torch.clip(noise + feats[:, :3], 0, 255)
        return coords, feats, labels


class HueSaturationTranslation():

    @staticmethod
    def rgb_to_hsv(rgb):
        # Translated from source of colorsys.rgb_to_hsv
        # r,g,b should be a numpy arrays with values between 0 and 255
        # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
        rgb = rgb.to('float')
        hsv = torch.zeros_like(rgb)
        # in case an RGBA array was passed, just copy the A channel
        hsv[..., 3:] = rgb[..., 3:]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = torch.max(rgb[..., :3], dim=-1)[0]
        minc = torch.min(rgb[..., :3], dim=-1)[0]
        hsv[..., 2] = maxc
        mask = maxc != minc
        hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
        rc = torch.zeros_like(r)
        gc = torch.zeros_like(g)
        bc = torch.zeros_like(b)
        rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
        gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
        bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
        hsv[..., 0] = np.select([r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
        hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
        return hsv

    @staticmethod
    def hsv_to_rgb(hsv):
        # Translated from source of colorsys.hsv_to_rgb
        # h,s should be a numpy arrays with values between 0.0 and 1.0
        # v should be a numpy array with values between 0.0 and 255.0
        # hsv_to_rgb returns an array of uints between 0 and 255.
        rgb = np.empty_like(hsv)
        rgb[..., 3:] = hsv[..., 3:]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = (h * 6.0).astype('uint8')
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
        rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
        rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
        rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
        return rgb.astype('uint8')

    def __init__(self, hue_max, saturation_max):
        raise NotImplementedError
        self.hue_max = hue_max
        self.saturation_max = saturation_max

    def __call__(self, coords, feats, labels):
        # Assume feat[:, :3] is rgb
        hsv = HueSaturationTranslation.rgb_to_hsv(feats[:, :3])
        hue_val = (random.random() - 0.5) * 2 * self.hue_max
        sat_ratio = 1 + (random.random() - 0.5) * 2 * self.saturation_max
        hsv[..., 0] = np.remainder(hue_val + hsv[..., 0] + 1, 1)
        hsv[..., 1] = np.clip(sat_ratio * hsv[..., 1], 0, 1)
        feats[:, :3] = np.clip(HueSaturationTranslation.hsv_to_rgb(hsv), 0, 255)

        return coords, feats, labels


##############################
# Coordinate transformations
##############################


class RandomDropout(AbstractTransform):

    def __init__(self, dropout_ratio=0.5, apply_ratio=0.2):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.dropout_ratio = dropout_ratio
        self.apply_ratio = apply_ratio

    def __call__(self, coords, feats, labels):
        if random.random() < self.apply_ratio:
            N = len(coords)
            inds = sorted(np.random.choice(
                N, int(N * (1 - self.dropout_ratio)), replace=False))
            return coords[inds], feats[inds], labels[inds]
        return coords, feats, labels


class RandomHorizontalFlip(AbstractTransform):

    def __init__(self, upright_axis, axis_ratio=[0.5, 0.5], apply_ratio=0.95):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.D = 3
        self.apply_ratio = apply_ratio
        self.axis_ratio = axis_ratio
        self.upright_axis = {'x': 0, 'y': 1, 'z': 2}[upright_axis.lower()]
        # Use the rest of axes for flipping.
        self.horz_axes = set(range(self.D)) - set([self.upright_axis])

    def __call__(self, coords, feats, labels):
        if random.random() < self.apply_ratio:
            for i, curr_ax in enumerate(self.horz_axes):
                if random.random() < self.axis_ratio[i]:
                    coord_max = torch.max(coords[:, curr_ax])
                    coords[:, curr_ax] = coord_max - coords[:, curr_ax]
        return coords, feats, labels


class ElasticDistortion(AbstractTransform):

    def __init__(self, distortion_params, apply_ratio=0.95):
        self.distortion_params = distortion_params
        self.apply_ratio = apply_ratio

    def elastic_distortion(self, coords, feats, labels, granularity, magnitude):
        """Apply elastic distortion on sparse coordinate space.

          pointcloud: numpy array of (number of points, at least 3 spatial dims)
          granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
          magnitude: noise multiplier
        """
        device = coords.device
        coords = coords.cpu()
        blurx = torch.ones((3, 1, 1, 1)).to(torch.float32) / 3
        blury = torch.ones((1, 3, 1, 1)).to(torch.float32) / 3
        blurz = torch.ones((1, 1, 3, 1)).to(torch.float32) / 3
        coords_min = coords.min(0)[0]

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = torch.div((coords - coords_min).max(0)[0], granularity, rounding_mode='floor').to(int) + 3
        noise = torch.randn(*noise_dim, 3).to(torch.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(noise, blurx, mode='constant', cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blury, mode='constant', cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blurz, mode='constant', cval=0)

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [torch.linspace(d_min, d_max, d) for d_min, d_max, d in zip(coords_min - granularity, coords_min + granularity * (noise_dim - 2), noise_dim)]
        interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=0, fill_value=0)
        # if granularity == 0.2:
        #     coords = Transform(coords)
        coords += interp(coords) * magnitude
        return coords.to(device), feats, labels

    def __call__(self, coords, feats, labels):
        if random.random() < self.apply_ratio:
            if self.distortion_params is not None:
                for granularity, magnitude in self.distortion_params:
                    coords, feats, labels = self.elastic_distortion(coords, feats, labels, granularity, magnitude)
        return coords, feats, labels


class Compose(AbstractTransform):
    """Composes several transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, coords, feats, labels, *args):
        for t in self.transforms:
            if t.COORD_DIM == coords.shape[1]:
                coords, feats, labels = t(coords, feats, labels)
            elif t.COORD_DIM == 3 and coords.shape[1] == 4:
                indices, slim_coords = coords[:, 0:1], coords[:, 1:]
                slim_coords, feats, labels = t(slim_coords, feats, labels)
                coords = torch.cat((indices, slim_coords), dim=1)
            else:
                raise NotImplementedError

        return coords, feats, labels, *args

class SplitCompose(object):
    def __init__(self, sync_transform, random_transform, coords_dim=3) -> None:
        self.sync_transform = sync_transform
        self.random_transform = random_transform
    
    def __call__(self, coords, feats, labels, *args):
        for t in self.sync_transform:
            coords, feats, labels = t(coords, feats, labels)
        
        coords_a, feats_a, labels_a = coords, feats, labels
        coords_b, feats_b, labels_b = coords.clone(), feats.clone(), labels.clone()

        for t in self.random_transform:
            if t.COORD_DIM == coords.shape[1]:
                coords_a, feats_a, labels_a = t(coords_a, feats_a, labels_a)
                coords_b, feats_b, labels_b = t(coords_b, feats_b, labels_b)
            elif t.COORD_DIM == 3 and coords.shape[1] == 4:
                indices_a, slim_coords_a = coords_a[:, 0:1], coords_a[:, 1:]
                slim_coords_a, feats_a, labels_a = t(slim_coords_a, feats_a, labels_a)
                coords_a = torch.cat((indices_a, slim_coords_a), dim=1)
                indices_b, slim_coords_b = coords_b[:, 0:1], coords_b[:, 1:]
                slim_coords_b, feats_b, labels_b = t(slim_coords_b, feats_b, labels_b)
                coords_b = torch.cat((indices_b, slim_coords_b), dim=1)
            else:
                raise NotImplementedError
        return (coords_a, feats_a, labels_a), (coords_b, feats_b, labels_b), *args


class Voxelize(AbstractTransform):

    def __init__(self, fix_map=None, voxel_size=0.05, return_map=False, apply_ratio=1.) -> None:
        self.fix_map = fix_map
        self.voxel_size = voxel_size
        self.return_map = return_map

    def __call__(self, coords, feats, labels):
        device = coords.device
        transformed_coords = coords / self.voxel_size
        if self.fix_map is None:
            voxelized_coords, voxelized_feats, voxelized_labels, indices_map, reverse_indices_map = ME.utils.sparse_quantize(
                transformed_coords.cpu().numpy(), feats.cpu().numpy(), labels.cpu().numpy(), return_index=True, return_inverse=True, ignore_label=255
            )
        else: 
            voxelized_coords, voxelized_feats, voxelized_labels, indices_map, reverse_indices_map = \
                transformed_coords[self.fix_map], feats[self.fix_map], labels[self.fix_map], self.fix_map, None
        if self.return_map:
            return torch.as_tensor(voxelized_coords, device=device), torch.as_tensor(voxelized_feats, device=device), torch.as_tensor(voxelized_labels, device=device), torch.as_tensor(indices_map, device=device), torch.as_tensor(reverse_indices_map, device=torch.DeviceObjType),
        else:
            return torch.as_tensor(voxelized_coords, device=device), torch.as_tensor(voxelized_feats, device=device), torch.as_tensor(voxelized_labels, device=device)


class RandomRotation(AbstractTransform):

    BASE_VEC = lambda _, x: np.array((1, 0, 0)) if x == 0 else (np.array((0, 1, 0)) if x == 1 else np.array((0, 0, 1)))

    def __init__(self, boundaries=[
        (-np.pi / 64, np.pi / 64),
        (-np.pi / 64, np.pi / 64),
        (-np.pi, np.pi),
        ], apply_ratio=1.) -> None:
        self.boundaries = boundaries
        self.apply_ratio = apply_ratio

    def rotate_mat(self, axis, theta, device):
        """
        Return rotation matrix alone certain `axis` by `angle`
        """
        return torch.tensor(expm(np.cross(np.eye(3), axis / norm(axis) * theta)), dtype=torch.float32, device=device)

    def __call__(self, coords, feats, labels):
        if random.random() < self.apply_ratio:
            rotation_mat = torch.eye(3, device=coords.device)
            for axis_index, boundary in enumerate(self.boundaries):
                theta = np.random.uniform(*boundary)
                axis = self.BASE_VEC(axis_index)
                rotation_mat = rotation_mat @ self.rotate_mat(axis, theta, coords.device)
            coords = coords @ rotation_mat.T
        return coords, feats, labels

class ToSparseTensor(AbstractTransform):
    COORD_DIM = 4
    def __init__(self) -> None:
        ...
    
    def __call__(self, coords, feats, labels):
        assert coords.device == feats.device
        tensor_field = ME.TensorField(
            coordinates=coords.int(),
            features=feats,
            # quantization_mode=ME.SparseTensorQuantizationMode.RANDOM_SUBSAMPLE,
            device=coords.device
        )
        return tensor_field, tensor_field.sparse(), labels

class ToDevice(AbstractTransform):
    def __init__(self, device) -> None:
        self.device = device
    
    def __call__(self, coords, feats, labels):
        return coords.to(self.device), feats.to(self.device), labels.to(self.device)


class RandomScaling(AbstractTransform):
    def __init__(self, boundaries=(
        0.9, 1.1
    ), apply_ratio=1.) -> None:
        self.boundaries = boundaries
        self.apply_ratio = apply_ratio
    def __call__(self, coords, feats, labels):
        if random.random() < self.apply_ratio:
            scale = np.random.uniform(*self.boundaries)
            coords = coords * scale
        return coords, feats, labels

class NonNegativeTranslation(AbstractTransform):
    def __init__(self, apply_ratio=1.) -> None:
        self.apply_ratio = apply_ratio

    def __call__(self, coords, feats, labels):
        if random.random() < self.apply_ratio:
            min_coord = coords.min(0)[0]
            coords = coords - min_coord
        return coords, feats, labels


class RandomTranslation(AbstractTransform):
    def __init__(self, boundaries=[(-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2)], apply_ratio=0.95) -> None:
        self.boundaries = boundaries
        self.apply_ratio = apply_ratio

    def __call__(self, coords, feats, labels):
        if random.random() < self.apply_ratio:
            scales = torch.tensor([np.random.uniform(*self.boundaries[i]) for i in range(3)], device=coords.device)
            min_coord = coords.min(0)[0]
            max_coord = coords.max(0)[0]

            coords = coords + scales * (max_coord - min_coord)
        return coords, feats, labels
        
class cf_collate_fn_factory:
    """Generates collate function for coords, feats, labels.

      Args:
        limit_numpoints: If 0 or False, does not alter batch size. If positive integer, limits batch
                         size so that the number of input coordinates is below limit_numpoints.
    """

    def __init__(self, limit_numpoints, device='cuda'):
        self.limit_numpoints = limit_numpoints
        self.device = device

    def __call__(self, list_data):
        coords_batch, feats_batch, labels_batch = [], [], []

        batch_id = 0
        batch_num_points = 0
        for batch_id, (coords, feats, labels, *_) in enumerate(list_data):
            coords = coords.to(coords.device)
            feats = feats.to(feats.device)
            labels = labels.to(labels.device)
            num_points = coords.shape[0]
            batch_num_points += num_points
            if self.limit_numpoints is not None and batch_num_points > self.limit_numpoints:
                num_full_points = sum(len(pack[0]) for pack in list_data)
                num_full_batch_size = len(list_data)
                logging.getLogger(__name__).warning(
                    f'Cannot fit {num_full_points} points into '
                    f'{self.limit_numpoints} points limit. '
                    f'Truncating batch size at {batch_id} '
                    f'out of {num_full_batch_size} with {batch_num_points - num_points}.',)
                break
            coords_batch.append(torch.cat((torch.ones(num_points, 1, device=coords.device).int() * batch_id, coords.int()), 1))
            feats_batch.append(feats)
            labels_batch.append(labels.int())

        # Concatenate all lists
        assert len(coords_batch) > 0, "limit_numpoints too low, not enough to load a single scene"
        coords_batch = torch.cat(coords_batch, 0).float()
        feats_batch = torch.cat(feats_batch, 0).float()
        labels_batch = torch.cat(labels_batch, 0).int()
        return coords_batch, feats_batch, labels_batch, *tuple(zip(*list_data))[3:]
