import numpy as np
import torch
from plyfile import PlyData
from torch.utils.data import Dataset


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
        device=None,
    ) -> None:
        super(PLYPointCloudDataset, self).__init__(data_list_path, label_list_path)
        self.return_paths = return_paths
        self.device = device

    def sample_loader(self, data_path, label_path=None):
        data = PlyData.read(data_path)

        # N * 3, float32
        coords = torch.as_tensor(np.stack((data['vertex']['x'], data['vertex']['y'], data['vertex']['z']), axis=1)).to(self.device)

        # N * 3, uint8
        feats = torch.as_tensor(np.stack((data['vertex']['red'], data['vertex']['green'], data['vertex']['blue']), axis=1)).to(self.device)

        ret = [coords, feats]

        if label_path is not None:
            labels = PlyData.read(label_path)
            labels_data = torch.as_tensor(labels['vertex']['label']).to(self.device)
            ret.append(labels_data)

        if self.return_paths:
            ret.append(data_path)
            ret.append(label_path)

        return ret


class ScanNetDataset(PLYPointCloudDataset):
    VALID_LABEL_IDS = [
        (1, "wall"),
        (2, "floor"),
        (3, "cabinet"),
        (4, "bed"),
        (5, "chair"),
        (6, "sofa"),
        (7, "table"),
        (8, "door"),
        (9, "window"),
        (10, "bookshelf"),
        (11, "picture"),
        (12, "counter"),
        (14, "desk"),
        (16, "curtain"),
        (24, "refridgerator"),
        (28, "shower curtain"),
        (33, "toilet"),
        (34, "sink"),
        (36, "bathtub"),
        (39, "otherfurniture"),
    ]

    def sample_loader(self, data_path, label_path=None):
        result = super().sample_loader(data_path, label_path)
        if self.label_paths is not None:
            for idx, (label_id, _) in enumerate(self.VALID_LABEL_IDS):
                result[2][result[2] == label_id] = idx
        return result

