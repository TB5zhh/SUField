# %%
import configparser
import os
from typing import *

import numpy as np
import open3d as o3d
import potpourri3d as pp3d
import pymeshlab
import torch
# from IPython import embed
from plyfile import PlyData, PlyElement
from sklearnex import patch_sklearn
from tqdm import tqdm

from utils import count_time, log, timer

SCANNET_COLOR_MAP = {
    0: (0., 0., 0.),
    1: (174., 199., 232.),
    2: (152., 223., 138.),
    3: (31., 119., 180.),
    4: (255., 187., 120.),
    5: (188., 189., 34.),
    6: (140., 86., 75.),
    7: (255., 152., 150.),
    8: (214., 39., 40.),
    9: (197., 176., 213.),
    10: (148., 103., 189.),
    11: (196., 156., 148.),
    12: (23., 190., 207.),
    13: (124., 232., 109.),
    14: (247., 182., 210.),
    15: (66., 188., 102.),
    16: (219., 219., 141.),
    17: (140., 57., 197.),
    18: (202., 185., 52.),
    19: (51., 176., 203.),
    20: (200., 54., 131.),
    21: (92., 193., 61.),
    22: (78., 71., 183.),
    23: (172., 114., 82.),
    24: (255., 127., 14.),
    25: (91., 163., 138.),
    26: (153., 98., 156.),
    27: (140., 153., 101.),
    28: (158., 218., 229.),
    29: (100., 125., 154.),
    30: (178., 127., 135.),
    31: (56., 23, 131.),
    32: (146., 111., 194.),
    33: (44., 160., 44.),
    34: (112., 128., 144.),
    35: (96., 207., 209.),
    36: (227., 119., 194.),
    37: (213., 92., 176.),
    38: (94., 106., 211.),
    39: (82., 84., 163.),
    40: (100., 85., 144.),
}

patch_sklearn()
from sklearn.cluster import KMeans
from IPython import embed
import wandb
from random import randint

CONF_FILE = '/home/aidrive/tb5zhh/3d_scene_understand/SUField/conf.ini'
QUIET = True
WANDB = False
"""
Terminology:
ply_name (scan_id): scene0000_00
ply_file: scene0000_00.ply
ply_path: [dir]/scene0000_00.ply
"""


# %%
# TODO test
def add_fields_online(plydata: PlyData, fields=[
    ('nx', 'double'),
    ('ny', 'double'),
    ('nz', 'double'),
], clear=True):
    p = plydata
    v, f = p.elements
    a = np.empty(len(v.data), v.data.dtype.descr + fields)
    for name in v.data.dtype.fields:
        a[name] = v[name]
    if clear:
        a[''] = 0
    # Recreate the PlyElement instance
    v = PlyElement.describe(a, 'vertex')

    # Recreate the PlyData instance
    p = PlyData([v, f], text=True)

    return p


# %%


def plydata_to_arrays(plydata: PlyData) -> Tuple[np.ndarray, np.ndarray]:
    vertices = np.vstack((plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z'])).T
    faces = np.stack(plydata['face']['vertex_indices'])
    return vertices, faces


def setup_mapping(ply_origin: PlyData, ply_down_sampled: PlyData):
    full_coords, _ = plydata_to_arrays(ply_origin)
    sampled_coords, _ = plydata_to_arrays(ply_down_sampled)
    full_coords = torch.as_tensor(full_coords).cuda()
    sampled_coords = torch.as_tensor(sampled_coords).cuda()
    full2sampled = torch.as_tensor([((sampled_coords - coord)**2).sum(dim=1).min(dim=0)[1] for coord in full_coords
                                   ])  # use index in full mesh to find index of the closest in sampled mesh
    sampled2full = torch.as_tensor([((full_coords - coord)**2).sum(dim=1).min(dim=0)[1] for coord in sampled_coords
                                   ])  # use index in smapled mesh to find index of the closest in full mesh
    del full_coords, sampled_coords
    return full2sampled, sampled2full


# %%
import re


class SpecClusterPipeline():

    def __init__(self, scan_path, count=200) -> None:
        self._load_plydata(scan_path)
        self.scan_id = re.findall('scene\d\d\d\d_\d\d', scan_path)[0]
        self.conf = configparser.ConfigParser()
        self.conf.read(CONF_FILE)
        self.conf = self.conf['Debug']
        self._load_sample_ids(count)

    def _load_plydata(self, scan_path):
        self.full_plydata = PlyData.read(scan_path)
        return self

    def _load_sample_ids(self, count=200):
        self.sample_ids: List[int] = []
        while len(self.sample_ids) < count:
            while True:
                drawn = randint(0, len(self.full_plydata['vertex']) - 1)
                if drawn not in self.sample_ids:
                    self.sample_ids.append(drawn)
                    break
        return self

    @timer
    @log(False)
    def downsample(self, dstarget=8000):
        assert hasattr(self, 'full_plydata') is not None
        print('start')
        # TODO restore color of the meshes
        temp_ply_path = f'/run/user/3023/.tmp_{randint(0,65535)}.ply'
        self.full_plydata.write(temp_ply_path)
        meshset = pymeshlab.MeshSet()
        meshset.load_new_mesh(temp_ply_path)

        meshset.apply_filter(
            'simplification_quadric_edge_collapse_decimation',
            targetperc=dstarget / len(self.full_plydata['vertex']),
            autoclean=True,
            qualitythr=0.8,
        )
        meshset.apply_filter('remove_unreferenced_vertices')

        meshset.save_current_mesh(temp_ply_path)
        new_ply_data = PlyData.read(temp_ply_path)
        os.remove(temp_ply_path)

        # TODO not necessary ?
        # new_ply_data = add_fields_online(new_ply_data, ['label', 'ushort'])
        self.sampled_plydata = new_ply_data
        print(f"downsampled size: {len(self.sampled_plydata['vertex'])}")
        return self

    @timer
    @log(QUIET)
    def setup_mapping(self):
        self.full2sampled, self.sampled2full = setup_mapping(self.full_plydata, self.sampled_plydata)
        return self

    @timer
    @log(QUIET)
    def calc_geod_dist(self):
        assert self.sampled_plydata is not None
        plydata = self.sampled_plydata
        with count_time('calculate geodesic distances'):
            vertices, faces = plydata_to_arrays(plydata)
            solver = pp3d.MeshHeatMethodDistanceSolver(vertices, faces)
            distances = []
            print(len(plydata['vertex']))
            for i in tqdm(range(len(plydata['vertex'])), disable=True):
                distances.append(solver.compute_distance(i))

            geod_mat = np.stack(distances)
            geod_mat = (geod_mat + geod_mat.T) / 2
        self.geod_mat = geod_mat
        return self

    @timer
    @log(QUIET)
    def calc_ang_dist(self, abs_inv=True, knn_range=None):
        assert self.sampled_plydata is not None
        knn = self.conf.getint('NormalKnnRange') if knn_range is None else knn_range

        pcd = o3d.geometry.PointCloud()
        vertices, _ = plydata_to_arrays(self.sampled_plydata)
        pcd.points = o3d.utility.Vector3dVector(vertices)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
        center = vertices.mean(axis=0)
        pcd.orient_normals_towards_camera_location(center)
        normals = torch.as_tensor(np.asarray(pcd.normals))
        # use absolute result
        if abs_inv:
            t = 1 - (normals @ normals.T).abs()
        else:
            t = 1 - (normals @ normals.T)
        ang_mat = t / t.mean()
        self.ang_mat: torch.Tensor = ang_mat
        return self

    @timer
    @log(QUIET)
    def calc_aff_mat(self, ratio: float = None):
        assert self.geod_mat is not None
        assert self.ang_mat is not None
        ratio = self.conf.getfloat('DistanceProportion') if ratio is None else ratio
        print(ratio)
        geod_mat = torch.as_tensor(self.geod_mat).cuda()
        ang_mat = self.ang_mat.cuda()
        dist_mat = ratio * geod_mat + (1 - ratio) * ang_mat
        dist_mat = (dist_mat + dist_mat.T) / 2
        sigma = dist_mat.mean() if self.conf.getboolean('AutoTemperature') else self.conf.getfloat('Temperature')
        aff_mat = np.e**(-dist_mat / (2 * sigma**2))

        aff_iv_mat = torch.diag(1 / aff_mat.sum(dim=1).sqrt())
        n_mat = (aff_iv_mat @ aff_mat @ aff_iv_mat)
        self.aff_mat: torch.Tensor = n_mat
        return self

    @timer
    @log(QUIET)
    def calc_embedding(self, feature: int = 30):
        assert self.aff_mat is not None
        eigh_vals, eigh_vecs = torch.linalg.eigh(self.aff_mat)
        del self.aff_mat
        eigh_vecs = eigh_vecs.T
        embedding = eigh_vecs[-feature:, :].flip(dims=(0,)).T
        embedding_cpu = embedding.cpu()
        self.embedding_mat = np.asarray(embedding_cpu)
        return self

    @timer
    @log(QUIET)
    def knn_cluster(self):
        shot = len(self.sample_ids)
        assert self.full_plydata is not None
        assert self.full2sampled is not None
        assert self.sample_ids is not None
        assert self.embedding_mat is not None
        selected_vertex_indices_in_sampled = self.full2sampled[self.sample_ids]
        selected_vertex_labels = self.full_plydata['vertex']['label'][self.sample_ids]
        self.cluster_result = KMeans(
            n_clusters=shot,
            init=self.embedding_mat[selected_vertex_indices_in_sampled],
        ).fit(self.embedding_mat)

        naive_indices = self.cluster_result.predict(self.embedding_mat)
        selected_predicted_labels = selected_vertex_labels[naive_indices]
        selected_predicted_distances = self.cluster_result.transform(self.embedding_mat)
        self.full_predicted_labels = selected_predicted_labels[self.full2sampled]
        full_naive_indices = naive_indices[self.full2sampled]
        self.full_predicted_distances = selected_predicted_distances[self.full2sampled]
        self.full_predicted_distances = self.full_predicted_distances[np.arange(len(self.full_predicted_distances)), full_naive_indices]
        print(self.full_predicted_distances.shape)
        return self

    def evaluate_cluster_result(self, correct_sum=None, total_sum=None):
        assert len(self.full_predicted_labels == self.full_plydata['vertex']['label'])
        correct = (self.full_plydata['vertex']['label'] == self.full_predicted_labels).sum()
        total = len(self.full_plydata['vertex'])
        if correct_sum is not None:
            correct_sum += correct
        if total_sum is not None:
            total_sum += total
        print(f'Correctness: {correct * 100/total:.2f}%')
        return self

    def evaluate_cluster_result_iou(self):
        Is = np.zeros((41))
        Os = np.zeros((41))
        for cls_idx in range(41):
            i = np.bitwise_and(self.full_predicted_labels == cls_idx, self.full_plydata['vertex']['label'] == cls_idx).sum()
            o = np.bitwise_or(self.full_predicted_labels == cls_idx, self.full_plydata['vertex']['label'] == cls_idx).sum()
            Is[cls_idx] = i
            Os[cls_idx] = o
        self.Is = Is
        self.Os = Os
        return self

    @log(QUIET)
    def save(self, save_dir="debug"):
        shot = len(self.sample_ids)
        os.makedirs(f"{save_dir}/spec_predictions", exist_ok=True)
        torch.save({
            "labels": self.full_predicted_labels,
            "confidence": self.full_predicted_distances,
        }, f"{save_dir}/spec_predictions/{self.scan_id}_{shot}.obj")
        return self

    def save_visualize(self, dir='debug'):
        assert self.full_predicted_labels is not None
        os.makedirs(dir, exist_ok=True)
        map_np = np.asarray(list(SCANNET_COLOR_MAP.values()))
        self.full_plydata['vertex']['red'] = map_np[:, 0][self.full_predicted_labels]
        self.full_plydata['vertex']['green'] = map_np[:, 1][self.full_predicted_labels]
        self.full_plydata['vertex']['blue'] = map_np[:, 2][self.full_predicted_labels]
        self.full_plydata.write(f'{dir}/{self.scan_id}.spec_clus.ply')
        return self


# %%
def main(scan_path, output_dir='debug', shot=200):
    pipeline = SpecClusterPipeline(scan_path, shot)
    pipeline \
        .downsample(dstarget=8000) \
        .calc_geod_dist() \
        .calc_ang_dist(abs_inv=True) \
        .calc_aff_mat(ratio=0.6) \
        .calc_embedding(feature=50) \
        .setup_mapping() \
        .knn_cluster() \
        .evaluate_cluster_result_iou() \
        .save_visualize(output_dir)
    # .save()


if __name__ == '__main__':
    main('/home/aidrive/tb5zhh/3d_scene_understand/SUField/data/scannetv2/scans/scene0001_00/scene0001_00_vh_clean_2.labels.ply')

# %%

# if __name__ == '__main__':
#     if WANDB:
#         wandb.init(project="spectral_cluster", entity="tb5zhh")
#     if len(sys.argv) == 1:
#         main((0, 1))
#     else:
#         main((int(sys.argv[1]), int(sys.argv[2])))
#     if WANDB:
#         print(wandb.run.name)

# VALID_CLASS_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)
# VALID_CLASS_IDS = list(VALID_CLASS_IDS)

# def collect():
#     """collect results from mid_Is_x.npy and mid_Os_x.npy"""
#     Is = np.zeros((4, 41))
#     Os = np.zeros((4, 41))
#     for i in range(8):
#         with open(f'another_mid_Is_{i}.npy', 'rb') as f:
#             Is += np.load(f)
#         with open(f'another_mid_Os_{i}.npy', 'rb') as f:
#             Os += np.load(f)
#     with open('tmp.out', 'w') as f:
#         for i in (Is / (Os + 1e-10))[:, VALID_CLASS_IDS]:
#             for j in i:
#                 print(j, end='\t', file=f)
#             print(file=f)
#     print((Is / (Os + 1e-10))[:, VALID_CLASS_IDS].mean(axis=1))

# %%
