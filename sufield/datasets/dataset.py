from logging import getLogger
from random import randint
from typing import Generic, TypeVar
import numpy as np
import torch
from plyfile import PlyData
from torch.utils.data import Dataset

from sufield.lib.utils.distributed import get_rank

from ..lib.utils.visualize import dump_points_with_labels

from . import transforms as t


class ToyDataset(Dataset):

    def __init__(self, length=100) -> None:
        self.len = length

    def __getitem__(self, index):
        return index

    def __len__(self):
        return self.len


class BundledDataset(Dataset):

    def __init__(self, bundle_path: str, **_) -> None:
        super().__init__()
        self.bundle = np.load(bundle_path, allow_pickle=True).item()
        self.coords = self.bundle['coords']
        self.feats = self.bundle['feats']
        if 'labels' in self.bundle.keys():
            self.labels = self.bundle['labels']
        else:
            self.labels = None

    def __getitem__(self, index):
        return torch.from_numpy(self.coords[index]), torch.from_numpy(self.feats[index]), torch.from_numpy(self.labels[index])

    def __len__(self):
        return self.bundle['size']


class FileListsDataset(Dataset):

    def __init__(self, data_list_path: str, label_list_path: str = None, **__) -> None:
        super().__init__()
        with open(data_list_path) as f:
            self.data_paths = [i.strip() for i in f.readlines()]
        if label_list_path is not None:
            with open(label_list_path) as f:
                self.label_paths = [i.strip() for i in f.readlines()]
        else:
            self.label_paths = None

    def sample_loader(self, data_path, label_path=None):
        """
        Given the path to a sample 
        and (optionally) the path of labels of the sample
        Load the sample deterministically
        """
        raise NotImplementedError

    def __getitem__(self, index):
        if self.label_paths is not None:
            return self.sample_loader(self.data_paths[index], self.label_paths[index])
        else:
            return self.sample_loader(self.data_paths[index])

    def __len__(self):
        return len(self.data_paths)


class PLYPointCloudDataset(FileListsDataset):

    def __init__(self, data_list_path: str, label_list_path: str = None, return_paths=False, **__) -> None:
        super(PLYPointCloudDataset, self).__init__(data_list_path, label_list_path)
        self.return_paths = return_paths

    def sample_loader(self, data_path, label_path=None):
        """
        Load sample from 'x', 'y', 'z', 'red', 'green', 'blue' field in a ply file
        (optionally) Load labels from 'label' field in a ply file

        Deterministic

        Return: coords, feats, labels (or None), data_path(optional), label_path(optional, if exists)
        """
        data = PlyData.read(data_path)
        # N * 3, float32
        coords = np.stack((data['vertex']['x'], data['vertex']['y'], data['vertex']['z']), axis=1)

        # N * 3, uint8
        feats = np.stack((data['vertex']['red'], data['vertex']['green'], data['vertex']['blue']), axis=1)

        ret = [torch.from_numpy(coords), torch.from_numpy(feats)]

        if label_path is not None:
            labels = PlyData.read(label_path)
            labels_data = labels['vertex']['label'].astype(np.uint8)
            ret.append(torch.from_numpy(labels_data))
        else:
            ret.append(None)

        if self.return_paths:
            ret.append(data_path)
            if label_path is not None:
                ret.append(label_path)

        return tuple(ret)


class Dynamic(Dataset):

    def __init__(self, cls, *args, **kwargs) -> None:
        self.base = cls(*args, **kwargs)

    def __getitem__(self, index):
        return self.base.__getitem__(index)

    def __len__(self):
        return len(self.base)


class ScanNet(Dynamic):
    """
    This dataset maps original label ids to label indices
    """
    TOTAL_CLS = 41
    VALID_LABEL_IDS = [(1, "wall"), (2, "floor"), (3, "cabinet"), (4, "bed"), (5, "chair"), (6, "sofa"), (7, "table"), (8, "door"), (9, "window"),
                       (10, "bookshelf"), (11, "picture"), (12, "counter"), (14, "desk"), (16, "curtain"), (24, "refridgerator"), (28, "shower curtain"),
                       (33, "toilet"), (34, "sink"), (36, "bathtub"), (39, "otherfurniture")]

    def __init__(self, cls, **kwargs) -> None:
        super().__init__(cls, **kwargs)

        # Define mapper from all labels to used labels
        # 1,2,3,5,6 -> 0,1,2,3,4
        self.MAPPER = [
            255 if class_idx not in [cls[0] for cls in self.VALID_LABEL_IDS] \
                else {cls[0]: idx for idx, cls in enumerate(self.VALID_LABEL_IDS)}[class_idx]
            for class_idx in range(self.TOTAL_CLS)
        ]

    def __getitem__(self, index):
        result = super().__getitem__(index)
        if result[2] is not None:
            for class_id in range(self.TOTAL_CLS):
                result[2][result[2] == class_id] = self.MAPPER[class_id]
        return result


# data_list_path: str, label_list_path: str = None, return_paths=False
class ScanNetVoxelized(ScanNet):

    def __init__(self, cls, **kwargs) -> None:
        super().__init__(cls, **kwargs)
        self.transforms = kwargs['transforms']

    def __getitem__(self, index):
        result = super().__getitem__(index)
        if self.transforms:
            result = self.transforms(*result)
        return result


class TrainValSplit(Dynamic):

    def __init__(self, cls, *args, **kwargs) -> None:
        super().__init__(cls, *args, **kwargs)
        split_file_dir = kwargs['split_file_dir']
        self.split = kwargs['split']
        with open(f"{split_file_dir}/scannetv2_train.txt") as f:
            train_list = [i.strip() for i in f.readlines()]
        with open(f"{split_file_dir}/scannetv2_val.txt") as f:
            val_list = [i.strip() for i in f.readlines()]
        with open(f"{split_file_dir}/scannetv2_test.txt") as f:
            test_list = [i.strip() for i in f.readlines()]
        full_list = sorted([*[(i, 0) for i in train_list], *[(i, 1) for i in val_list]], key=lambda x: x[0])
        if self.split == 'train':
            self.idx_map = [idx for idx, i in enumerate(full_list) if i[1] == 0]
            self.length = len(train_list)
        elif self.split == 'val':
            self.idx_map = [idx for idx, i in enumerate(full_list) if i[1] == 1]
            self.length = len(val_list)
        elif self.split == 'test':
            self.idx_map = [i for i in range(len(test_list))]
            self.length = len(test_list)

    def __getitem__(self, index):
        return super().__getitem__(self.idx_map[index])

    def __len__(self):
        return self.length


class LimitedTrainValSplit(TrainValSplit):

    def __init__(self, cls, *args, **kwargs) -> None:
        super().__init__(cls, *args, **kwargs)
        self.split_dir = kwargs['annotate_idx_dir']
        self.limited_size = int(kwargs['size'])
        self.map_idx = int(kwargs['map_idx']) if 'map_idx' in kwargs.keys() else None
        self.label_indices = torch.load(f"{self.split_dir}/points{self.limited_size}")
        self.indices = [(k, v) for k, v in sorted(self.label_indices.items(), key=lambda i: i[0])]

    def __getitem__(self, index):
        result = super().__getitem__(index)
        if self.split == 'train' and result[2] is not None:
            label_copy = torch.clone(result[2])
            result[2].fill_(255)
            indices = result[self.map_idx][1][self.indices[index][1]] if self.map_idx is not None else self.indices[index][1]
            result[2][indices] = label_copy[indices]
        return result


from IPython import embed


def test():
    transforms = t.Compose([
        t.ToTensor(),
        # t.ElasticDistortion(((0.2, 0.4), (0.8, 1.6))),
        # t.RandomRotation(((0,0),(0,0),(np.pi/2, np.pi/2))),
        # t.RandomScaling(),
        # t.NonNegativeTranslation()
        # t.Voxelize(),
        # t.RandomDropout(0.5, 1),
        # t.RandomTranslation([(2, 2), (2, 2), (2, 2)], 1),
        # t.ChromaticAutoContrast(),
        # t.ChromaticTranslation(0.1),
        # t.ChromaticJitter(0.05),
        # t.HueSaturationTranslation(0.5, 0.2)
    ])
    # dataset = ScanNetVoxelized(
    #     BundledDataset,
    #     bundle_path='/home/tb5zhh/SUField/bundled_datasets/SCANNET/train.npy',
    #     transforms=transforms,
    # )
    dataset = LimitedTrainValSplit(
        ScanNetVoxelized,
        BundledDataset,
        bundle_path='/home/tb5zhh/SUField/bundled_datasets/SCANNET/train.npy',
        transforms=transforms,
        annotate_idx_dir='/home/tb5zhh/SUField/bundled_datasets/SCANNET/limited_annotation',
        size=200,
        split_file_dir='/home/tb5zhh/SUField/bundled_datasets/SCANNET/split',
        split='train',
    )
    with open('/home/tb5zhh/SUField/bundled_datasets/SCANNET/split/scannetv2_train.txt') as f:
        l = sorted([i.strip() for i in f.readlines()])
    for idx in range(1201):
        result = dataset[idx]
        print(l[idx])
        print((result[2]!=255).sum())
        # dump_points_with_labels(result[0], result[2], 'voxelized.txt')


if __name__ == '__main__':
    test()
