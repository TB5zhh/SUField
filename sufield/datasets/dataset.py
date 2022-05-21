import numpy as np
from plyfile import PlyData
from torch.utils.data import Dataset

from sufield.lib.visualize import dump_points_with_labels

from . import transforms as t


class FileListsDataset(Dataset):

    def __init__(self, data_list_path: str, label_list_path: str = None) -> None:
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

    def __init__(
        self,
        data_list_path: str,
        label_list_path: str = None,
        return_paths=False,
    ) -> None:
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

        ret = [coords, feats]

        if label_path is not None:
            labels = PlyData.read(label_path)
            labels_data = labels['vertex']['label'].astype(np.uint8)
            ret.append(labels_data)
        else:
            ret.append(None)

        if self.return_paths:
            ret.append(data_path)
            if label_path is not None:
                ret.append(label_path)

        return tuple(ret)


class ScanNetDataset(PLYPointCloudDataset):
    """
    This dataset maps original label ids to label indices
    """
    TOTAL_CLS = 41
    VALID_LABEL_IDS = [(1, "wall"), (2, "floor"), (3, "cabinet"), (4, "bed"), (5, "chair"), (6, "sofa"), (7, "table"),
                       (8, "door"), (9, "window"), (10, "bookshelf"), (11, "picture"), (12, "counter"), (14, "desk"),
                       (16, "curtain"), (24, "refridgerator"), (28, "shower curtain"), (33, "toilet"), (34, "sink"),
                       (36, "bathtub"), (39, "otherfurniture")]

    def __init__(self, data_list_path: str, label_list_path: str = None, return_paths=False) -> None:
        super().__init__(data_list_path, label_list_path, return_paths)

        # Define mapper from all labels to used labels
        # 1,2,3,5,6 -> 0,1,2,3,4
        self.MAPPER = [
            255 if class_idx not in [cls[0] for cls in self.VALID_LABEL_IDS] \
                else {cls[0]: idx for idx, cls in enumerate(self.VALID_LABEL_IDS)}[class_idx]
            for class_idx in range(self.TOTAL_CLS)
        ]

    def sample_loader(self, data_path, label_path=None):
        result = super().sample_loader(data_path, label_path)
        if self.label_paths is not None:
            for class_id in range(self.TOTAL_CLS):
                result[2][result[2] == class_id] = self.MAPPER[class_id]
        return result



class ScanNetVoxelizedDataset(ScanNetDataset):

    def __init__(
        self,
        data_list_path: str,
        label_list_path: str = None,
        return_paths=False,
        transforms=None,
    ) -> None:
        super().__init__(data_list_path, label_list_path, return_paths)
        self.transforms = transforms

    def sample_loader(self, data_path, label_path=None):
        result = super().sample_loader(data_path, label_path)
        if self.transforms:
            result = self.transforms(*result)
        return result


from IPython import embed


def test():
    dataset = ScanNetVoxelizedDataset(
        '/home/tb5zhh/SUField/datasets/ScanNetv2_train_data.txt',
        '/home/tb5zhh/SUField/datasets/ScanNetv2_train_labels.txt',
        return_paths=True,
        transforms=t.Compose([
            t.ToTensor(),
            # t.ElasticDistortion(((0.2, 0.4), (0.8, 1.6))),
            # t.RandomRotation(((0,0),(0,0),(np.pi/2, np.pi/2))),
            # t.RandomScaling(),
            # t.NonNegativeTranslation()
            # t.Voxelize(),
            # t.RandomDropout(0.5, 1),
            t.RandomTranslation([(2,2),(2,2),(2,2)], 1),
            # t.ChromaticAutoContrast(),
            # t.ChromaticTranslation(0.1),
            # t.ChromaticJitter(0.05),
            # t.HueSaturationTranslation(0.5, 0.2)
        ]),
    )
    result = dataset[0]
    dump_points_with_labels(result[0], result[2], 'voxelized.txt')


if __name__ == '__main__':
    test()