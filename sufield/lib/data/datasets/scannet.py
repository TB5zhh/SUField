import logging
import os
import sys
from pathlib import Path

import numpy as np
from lib.data.dataset import VoxelizedDataset, VoxelizedDatasetBase, VoxelizedTestDataset
from lib.pc_utils import read_plyfile, save_point_cloud
from lib.utils import fast_hist, per_class_iu, read_txt
from scipy import spatial
from sufield.config import CLASS_LABELS, SCANNET_COLOR_MAP, VALID_CLASS_IDS


class ScannetVoxelizedDatasetBase(VoxelizedDatasetBase):
    '''
    Labels properties
    '''
    NUM_LABELS_ALL = 41
    IGNORE_LABELS = tuple(set(range(41)) - set(VALID_CLASS_IDS))
    '''
    Prevoxelization tranformation parameters
    '''
    ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))
    '''
    Input transformation parameters
    '''
    ROTATION_AXIS = 'z'
    '''
    Voxelization parameters
    '''
    VOXEL_SIZE = 0.05
    CLIP_BOUND = None  # Do not clip
    SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
    TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
    ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi, np.pi))

    # If trainval.txt does not exist, copy train.txt and add contents from val.txt
    DATA_NAME_LIST = {
        'train': 'scannetv2_train.txt',
        'val': 'scannetv2_val.txt',
        'trainval': 'scannetv2_trainval.txt',
        'test': 'scannetv2_test.txt',
    }

    def __init__(self, config, split='train', **kwargs):

        self.data_root = None
        data_paths = read_txt(os.path.join('./splits/scannet', self.DATA_NAME_LIST[split]))

        logging.info('Loading {}: {}'.format(self.__class__.__name__, self.DATA_NAME_LIST[split]))
        super().__init__(
            data_root=self.data_root,
            data_paths=data_paths,
            config=config,
            return_transformation=config.return_transformation,
            ignore_mask=config.ignore_label,
            **kwargs,
        )

    def get_output_id(self, iteration):
        return '_'.join(Path(self.data_paths[iteration]).stem.split('_')[:2])

    def test_pointcloud(self, pred_dir):
        '''
        TODO remove this method
        '''
        print('Running full pointcloud evaluation.')
        eval_path = os.path.join(pred_dir, 'fulleval')
        os.makedirs(eval_path, exist_ok=True)
        # Join room by their area and room id.
        # Test independently for each room.
        sys.setrecursionlimit(100000)  # Increase recursion limit for k-d tree.
        hist = np.zeros((self.NUM_LABELS, self.NUM_LABELS))
        for i, data_path in enumerate(self.data_paths):
            room_id = self.get_output_id(i)
            pred = np.load(os.path.join(pred_dir, 'pred_%04d_%02d.npy' % (i, 0)))

            # save voxelized pointcloud predictions
            save_point_cloud(np.hstack((pred[:, :3], np.array([SCANNET_COLOR_MAP[i] for i in pred[:, -1]]))), f'{eval_path}/{room_id}_voxel.ply', verbose=False)

            fullply_f = self.data_root / data_path
            query_pointcloud = read_plyfile(fullply_f)
            query_xyz = query_pointcloud[:, :3]
            query_label = query_pointcloud[:, -1]
            # Run test for each room.
            pred_tree = spatial.KDTree(pred[:, :3], leafsize=500)
            _, result = pred_tree.query(query_xyz)
            ptc_pred = pred[result, 3].astype(int)
            # Save prediciton in txt format for submission.
            np.savetxt(f'{eval_path}/{room_id}.txt', ptc_pred, fmt='%i')
            # Save prediciton in colored pointcloud for visualization.
            save_point_cloud(np.hstack((query_xyz, np.array([SCANNET_COLOR_MAP[i] for i in ptc_pred]))), f'{eval_path}/{room_id}.ply', verbose=False)
            # Evaluate IoU.
            if self.IGNORE_LABELS is not None:
                ptc_pred = np.array([self.label_map[x] for x in ptc_pred], dtype=np.int)
                query_label = np.array([self.label_map[x] for x in query_label], dtype=np.int)
            hist += fast_hist(ptc_pred, query_label, self.NUM_LABELS)
        ious = per_class_iu(hist) * 100
        print('mIoU: ' + str(np.nanmean(ious)) + '\n' 'Class names: ' + ', '.join(CLASS_LABELS) + '\n' 'IoU: ' + ', '.join(np.round(ious, 2).astype(str)))


class ScannetVoxelizedDataset(ScannetVoxelizedDatasetBase, VoxelizedDataset):

    def __init__(self, config, **kwargs):
        self.data_root = config.scannet_path
        super().__init__(config, **kwargs)


class ScannetVoxelizedTestDataset(ScannetVoxelizedDatasetBase, VoxelizedTestDataset):

    def __init__(self, config, **kwargs):
        self.data_root = config.scannet_test_path
        super().__init__(config, **kwargs)


class ScannetVoxelization2cmDataset(ScannetVoxelizedDataset):
    VOXEL_SIZE = 0.02


class ScannetVoxelization2cmDataset(ScannetVoxelizedTestDataset):
    VOXEL_SIZE = 0.02
