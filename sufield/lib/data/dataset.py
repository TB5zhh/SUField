import math
from abc import ABC
from pathlib import Path

import MinkowskiEngine as ME
import numpy as np
import sufield.lib.transforms as t
import torch
from plyfile import PlyData
from sufield.lib.data.sampler import DistributedInfSampler, InfSampler
from sufield.lib.data.voxelizer import TestVoxelizer, Voxelizer
from sufield.lib.distributed_utils import get_world_size
from torch.utils.data import DataLoader, Dataset


def cache(func):

    def wrapper(self, *args, **kwargs):
        # Assume that args[0] is index
        index = args[0]
        if self.cache:
            if index not in self.cache_dict[func.__name__]:
                results = func(self, *args, **kwargs)
                self.cache_dict[func.__name__][index] = results
            return self.cache_dict[func.__name__][index]
        else:
            return func(self, *args, **kwargs)

    return wrapper


class DictDataset(Dataset, ABC):

    # IS_FULL_POINTCLOUD_EVAL = False

    def __init__(self, data_root, data_paths):
        """
        data_paths: list of lists, [[str_path_to_input, str_path_to_label], [...]]
        """
        Dataset.__init__(self)

        if not isinstance(data_root, Path):
            data_root = Path(data_root)
        self.data_root = data_root
        self.data_paths = sorted(data_paths)

        # TODO check usage of cache

    def get_classnames(self):
        pass

    def reorder_result(self, result):
        return result

    def __len__(self):
        return len(self.data_paths)


class VoxelizedDatasetBase(DictDataset, ABC):
    '''
    Labels properties
    '''
    NUM_LABELS_ALL = -1  # Number of labels in the dataset, including all ignore classes
    NUM_LABELS = -1  # Number of labels in the dataset
    IGNORE_LABELS = None  # List of labels that are not evaluated
    '''
    Prevoxelization tranformation parameters
    Downsample the pointcloud with finer voxel size before transformation for memory and speed
    '''
    ELASTIC_DISTORT_PARAMS = None
    '''
    Input transformation parameters
    '''
    ROTATION_AXIS = None
    '''
    Voxelization parameters
    TODO move these to config files
    1. prevoxelization voxelization
    2. clipping
    3. random scaling (include center random translation)
    4. random rotation
    5. sparse_quantize
    6. translation to positive coordinates
    '''
    VOXELIZER_CLS = None
    PREVOXELIZATION_VOXEL_SIZE = None
    VOXEL_SIZE = 0.05  # 5cm
    CLIP_BOUND = None
    SCALE_AUGMENTATION_BOUND = None
    TRANSLATION_AUGMENTATION_RATIO_BOUND = None
    ROTATION_AUGMENTATION_BOUND = None
    '''
    MISC
    '''
    IS_TEMPORAL = False  # Legacy
    AUGMENT_COORDS_TO_FEATS = False  # Concatenate feats and normalized coords to form new feats

    # NUM_IN_CHANNEL = None

    def __init__(self,
                 data_root,
                 data_paths,
                 prevoxel_transform=None,
                 input_transform=None,
                 ignore_mask=255,
                 return_transformation=False,
                 augment_data=False,
                 config=None):
        """
        ignore_mask: label value for ignore class. It will not be used as a class in the loss or evaluation.
        """
        DictDataset.__init__(self, data_root=data_root, data_paths=data_paths)

        self.prevoxel_transform = prevoxel_transform
        self.input_transform = input_transform
        self.ignore_mask = ignore_mask
        self.return_transformation = return_transformation
        self.augment_data = augment_data
        self.config = config

        self.voxelizer = self.VOXELIZER_CLS(voxel_size=self.VOXEL_SIZE,
                                            clip_bound=self.CLIP_BOUND,
                                            use_augmentation=augment_data,
                                            scale_augmentation_bound=self.SCALE_AUGMENTATION_BOUND,
                                            rotation_augmentation_bound=self.ROTATION_AUGMENTATION_BOUND,
                                            translation_augmentation_ratio_bound=self.TRANSLATION_AUGMENTATION_RATIO_BOUND,
                                            ignore_mask=ignore_mask)
        '''
        Construct self.label_map and self.NUM_LABELS
        label_map: (0, 1, 2, 3, 4, 5, ... n, unused) --mapped--> (0, unused, unused, 1, 2, unused, ... m, unused)
        '''
        label_map = {}
        n_used = 0
        for idx in range(self.NUM_LABELS_ALL):
            if idx in self.IGNORE_LABELS:
                label_map[idx] = self.ignore_mask
            else:
                label_map[idx] = n_used
                n_used += 1
        label_map[self.ignore_mask] = self.ignore_mask
        self.label_map = label_map
        self.NUM_LABELS -= len(self.IGNORE_LABELS)

    def __getitem__(self, index):
        raise NotImplementedError

    def load_ply(self, index):
        filepath = self.data_root / self.data_paths[index]
        plydata = PlyData.read(filepath)
        data = plydata.elements[0].data
        coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
        feats = np.array([data['red'], data['green'], data['blue']], dtype=np.float32).T
        labels = np.array(data['label'], dtype=np.int32)
        return coords, feats, labels, None

    def _augment_coords_to_feats(self, coords, feats, labels=None):
        norm_coords = coords - coords.mean(0)
        # color must come first.
        if isinstance(coords, np.ndarray):
            feats = np.concatenate((feats, norm_coords), 1)
        else:
            feats = torch.cat((feats, norm_coords), 1)
        return coords, feats, labels


class VoxelizedTestDataset(VoxelizedDatasetBase):
    """
    This dataset loads RGB point clouds and their labels as a list of points
    and voxelizes the pointcloud with sufficient data augmentation.
    """
    VOXELIZER_CLS = TestVoxelizer

    def __getitem__(self, index):
        coords, feats, labels, center = self.load_ply(index)
        # Downsample the pointcloud with finer voxel size before transformation for memory and speed
        if self.PREVOXELIZATION_VOXEL_SIZE is not None:
            inds = ME.utils.sparse_quantize(coords / self.PREVOXELIZATION_VOXEL_SIZE, return_index=True)
            coords, feats, labels = coords[inds], feats[inds], labels[inds]

        # Prevoxel transformations
        if self.prevoxel_transform is not None:
            coords, feats, labels = self.prevoxel_transform(coords, feats, labels)

        coords, feats, labels, transformation = self.voxelizer.voxelize(coords, feats, labels, center=center)
        coords = coords.numpy()

        if self.input_transform is not None:
            coords, feats, labels = self.input_transform(coords, feats, labels)

        # Map gt labels to indices
        if self.IGNORE_LABELS is not None:
            labels = np.array([self.label_map[x] for x in labels], dtype=np.int)

        # Use coordinate features if config is set
        if self.AUGMENT_COORDS_TO_FEATS:
            coords, feats, labels = self._augment_coords_to_feats(coords, feats, labels)

        return_args = [coords, feats, labels]
        if self.return_transformation:
            return_args.append(transformation.astype(np.float32))

        return tuple(return_args)


class VoxelizedDataset(VoxelizedDatasetBase):
    """
    This dataset loads RGB point clouds and their labels as a list of points
    and voxelizes the pointcloud with sufficient data augmentation.
    """
    VOXELIZER_CLS = Voxelizer

    def Transform(self, a):
        scale = 20
        m = np.eye(3) + np.random.randn(3, 3) * 0.1
        m[0][0] *= np.random.randint(0, 2) * 2 - 1
        m *= scale
        theta = np.random.rand() * 2 * math.pi
        m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
        a = np.matmul(a, m)

        return a

    def __getitem__(self, index):
        coords, feats, labels, center = self.load_ply(index)
        if self.PREVOXELIZATION_VOXEL_SIZE is not None:
            inds = ME.utils.sparse_quantize(coords / self.voxel_size, return_index=True)
            coords, feats, labels = coords[inds], feats[inds], labels[inds]
            # Prevoxel transformations
        if self.prevoxel_transform is not None:
            coords1, feats1, labels1 = self.prevoxel_transform(coords, feats, labels)
            coords2, feats2, labels2 = self.prevoxel_transform(coords, feats, labels)

        coords1 = (self.Transform(coords1)).astype(np.int)
        coords2 = (self.Transform(coords2)).astype(np.int)

        coords1, feats1, labels1, coords2, feats2, labels2 = self.voxelizer.voxelize(coords1, coords2, feats1, labels1, feats2, labels2, center=center)

        # map labels not used for evaluation to ignore_label
        if self.input_transform is not None:
            coords1, feats1, labels1 = self.input_transform(coords1, feats1, labels1)
            coords2, feats2, labels2 = self.input_transform(coords2, feats2, labels2)
        if self.IGNORE_LABELS is not None:
            labels1 = np.array([self.label_map[x] for x in labels1], dtype=np.int)
            labels2 = np.array([self.label_map[x] for x in labels2], dtype=np.int)

        # Use coordinate features if config is set
        if self.AUGMENT_COORDS_TO_FEATS:
            coords1, feats1, labels1 = self._augment_coords_to_feats(coords1, feats1, labels1)
            coords2, feats2, labels2 = self._augment_coords_to_feats(coords2, feats2, labels2)

        return_args = [coords1, feats1, labels1, coords2, feats2, labels2]

        return tuple(return_args)


def initialize_data_loader(DatasetClass, config, split, num_workers, shuffle, repeat, augment_data, batch_size, limit_numpoints, input_transform=[]):
    """
    prevoxel_transforms: tranformation applied before voxelization
    """
    ###### Collate functions ######
    if config.return_transformation:
        collate_fn = t.cflt_collate_fn_factory(limit_numpoints)
    else:
        if augment_data:
            collate_fn = t.cfl_collate_fn_factory(limit_numpoints)
        else:
            collate_fn = t.cf_collate_fn_factory(limit_numpoints)

    ###### Pointcloud transformations ######
    if augment_data:
        prevoxel_transforms = t.Compose([t.ElasticDistortion(DatasetClass.ELASTIC_DISTORT_PARAMS)])
        input_transforms = t.Compose([
            *input_transform,
            t.RandomDropout(0.2),
            t.RandomHorizontalFlip(DatasetClass.ROTATION_AXIS, DatasetClass.IS_TEMPORAL),
            t.ChromaticAutoContrast(),
            t.ChromaticTranslation(config.data_aug_color_trans_ratio),
            t.ChromaticJitter(config.data_aug_color_jitter_std),
            # t.HueSaturationTranslation(config.data_aug_hue_max, config.data_aug_saturation_max),
        ])
    else:
        prevoxel_transforms = None
        input_transforms = None

    ###### Construct dataset and dataloader ######
    dataset = DatasetClass(
        config,
        prevoxel_transform=prevoxel_transforms,
        input_transform=input_transforms,
        augment_data=augment_data,
        split=split,
    )

    data_args = {
        'dataset': dataset,
        'num_workers': num_workers,
        'batch_size': batch_size,
        'collate_fn': collate_fn,
    }

    if repeat:
        if get_world_size() > 1:
            data_args['sampler'] = DistributedInfSampler(dataset, shuffle=shuffle)  # torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            data_args['sampler'] = InfSampler(dataset, shuffle)
    else:
        data_args['shuffle'] = shuffle

    data_loader = DataLoader(**data_args)

    return data_loader
