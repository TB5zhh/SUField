# %%
import builtins
import configparser
from faulthandler import disable
import os
from datetime import datetime
from typing import *

import numpy as np
import open3d as o3d
import potpourri3d as pp3d
import pymeshlab
import torch
from IPython import embed
from plyfile import PlyData, PlyElement
from sklearnex import patch_sklearn
from tqdm import tqdm

patch_sklearn()
from sklearn.cluster import KMeans

CONF_FILE = '/home/aidrive/tb5zhh/3d_scene_understand/SUField/conf.ini'
DATA_BASE_DIR = '/home/aidrive/tb5zhh/3d_scene_understand/SUField/data/scannetv2/scans'
SAMPLE_IDS_DIR = '/home/aidrive/tb5zhh/3d_scene_understand/SpecCluster/tbw/indices'
SAVE_DIR = '/home/aidrive/tb5zhh/3d_scene_understand/SUField/results'
QUIET = True
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
from contextlib import contextmanager
import functools


@contextmanager
def count_time(name=None):
    print(f"Process {(name+' ') if name is not None else ''}start")
    start = datetime.now()
    yield
    end = datetime.now()
    print(f"Process {(name+' ') if name is not None else ''}spent: {(end - start).seconds}s {(end-start).microseconds // 1000} ms")


def log(quiet=False):

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kw):
            a = builtins.print
            if not quiet:
                print(f"Called {func.__name__}()")
            else:
                builtins.print = lambda *kw, **args: ...
            ret = func(*args, **kw)
            builtins.print = a
            return ret

        return wrapper

    return decorator


def timer(fn):

    @functools.wraps(fn)
    def inner(*args, **kw):
        start = datetime.now()
        r = fn(*args, **kw)
        end = datetime.now()
        print(f"{fn.__name__} spent: {(end - start).seconds}s {(end-start).microseconds} ms")
        return r

    return inner


def plydata_to_arrays(plydata: PlyData) -> Tuple[np.ndarray, np.ndarray]:
    vertices = np.vstack((plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z'])).T
    faces = np.stack(plydata['face']['vertex_indices'])
    return vertices, faces


def setup_mapping(ply_origin: PlyData, ply_down_sampled: PlyData) -> Tuple[Dict, Dict]:
    full_coords, _ = plydata_to_arrays(ply_origin)
    sampled_coords, _ = plydata_to_arrays(ply_down_sampled)
    full_coords = torch.as_tensor(full_coords)
    sampled_coords = torch.as_tensor(sampled_coords)
    full2sampled = None
    full2sampled = np.array(torch.as_tensor([((sampled_coords - coord)**2).sum(dim=1).min(dim=0)[1] for coord in full_coords
                                            ]).cpu())  # use index in full mesh to find index of the closest in sampled mesh
    sampled2full = np.array(torch.as_tensor([((full_coords - coord)**2).sum(dim=1).min(dim=0)[1] for coord in sampled_coords
                                            ]).cpu())  # use index in smapled mesh to find index of the closest in full mesh
    del full_coords, sampled_coords
    return full2sampled, sampled2full


# %%


class SpecClusterPipeline():

    def __init__(self, scan_id: str, quiet=False) -> None:
        self.scan_id = scan_id
        self.quiet = quiet
        self.conf = configparser.ConfigParser()
        self.conf.read(CONF_FILE)
        self.conf = self.conf['Debug']
        self._load_sample_ids()
        self._load_plydata()

    def _load_sample_ids(self):
        f20 = torch.load(f"{SAMPLE_IDS_DIR}/20.dict")[self.scan_id]
        f50 = torch.load(f"{SAMPLE_IDS_DIR}/50.dict")[self.scan_id]
        f100 = torch.load(f"{SAMPLE_IDS_DIR}/100.dict")[self.scan_id]
        f200 = torch.load(f"{SAMPLE_IDS_DIR}/200.dict")[self.scan_id]
        self.sample_ids = {20: f20, 50: f50, 100: f100, 200: f200}
        return self

    def _load_plydata(self):
        self.full_plydata = PlyData.read(f'{DATA_BASE_DIR}/{self.scan_id}/{self.scan_id}_vh_clean_2.labels.ply')
        return self

    @log(QUIET)
    def downsample(self):
        assert hasattr(self, 'full_plydata') is not None
        print('start')
        # TODO restore color of the meshes
        temp_ply_path = f'.tmp_{self.scan_id}.ply'

        self.full_plydata.write(temp_ply_path)
        meshset = pymeshlab.MeshSet()
        meshset.load_new_mesh(temp_ply_path)

        meshset.apply_filter(
            'simplification_quadric_edge_collapse_decimation',
            targetfacenum=self.conf.getint('TargetFaceNum'),
            qualitythr=self.conf.getfloat('QualityThreshold'),
            preserveboundary=True,
            preservenormal=True,
            preservetopology=True,
        )
        meshset.apply_filter('remove_unreferenced_vertices')

        meshset.save_current_mesh(temp_ply_path)
        new_ply_data = PlyData.read(temp_ply_path)
        os.remove(temp_ply_path)

        # TODO not necessary ?
        # new_ply_data = add_fields_online(new_ply_data, ['label', 'ushort'])
        self.sampled_plydata = new_ply_data
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
            for i in tqdm(range(len(plydata['vertex'])), disable=QUIET):
                distances.append(solver.compute_distance(i))

            geod_mat = np.stack(distances)
            geod_mat = (geod_mat + geod_mat.T) / 2
        self.geod_mat = geod_mat
        return self

    @log(QUIET)
    def calc_ang_dist(self, normal_inv=True):
        assert self.sampled_plydata is not None
        pcd = o3d.geometry.PointCloud()
        vertices, _ = plydata_to_arrays(self.sampled_plydata)
        pcd.points = o3d.utility.Vector3dVector(vertices)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=self.conf.getint('NormalKnnRange')))
        center = vertices.mean(axis=0)
        pcd.orient_normals_towards_camera_location(center)
        normals = torch.as_tensor(np.array(pcd.normals))
        # use absolute result
        if normal_inv:
            t = 1 - (normals @ normals.T).abs()
        else:
            t = 1 - (normals @ normals.T)
        ang_mat = t / t.mean()
        self.ang_mat: torch.Tensor = ang_mat
        return self

    @log(QUIET)
    def calc_aff_mat(self, ratio: float = None):
        assert self.geod_mat is not None
        assert self.ang_mat is not None
        geod_mat = torch.as_tensor(self.geod_mat)
        ang_mat = self.ang_mat
        ratio = self.conf.getfloat('DistanceProportion') if ratio is None else ratio
        dist_mat = ratio * geod_mat + (1 - ratio) * ang_mat
        dist_mat = (dist_mat + dist_mat.T) / 2
        sigma = dist_mat.mean() if self.conf.getboolean('AutoTemperature') else self.conf.getfloat('Temperature')
        aff_mat = np.e**(-dist_mat / (2 * sigma**2))

        aff_iv_mat = torch.diag(1 / aff_mat.sum(dim=1).sqrt())
        n_mat = (aff_iv_mat @ aff_mat @ aff_iv_mat)
        self.aff_mat: torch.Tensor = n_mat
        return self

    @log(QUIET)
    def calc_embedding(self, feature: int):
        assert self.aff_mat is not None
        aff_mat = self.aff_mat
        eigh_vals, eigh_vecs = torch.linalg.eigh(aff_mat.cuda())
        eigh_vecs = eigh_vecs.T
        embedding = eigh_vecs[-feature:, :].flip(dims=(0,)).T
        embedding_cpu = embedding.cpu()
        self.embedding_mat = np.array(embedding_cpu)
        return self

    @log(QUIET)
    def knn_cluster(self, shot: int):
        assert shot in (20, 50, 100, 200)
        assert self.full_plydata is not None
        assert self.full2sampled is not None
        assert self.sample_ids is not None
        assert self.embedding_mat is not None
        selected_vertex_indices_in_sampled = self.full2sampled[self.sample_ids[shot]]
        selected_vertex_labels = self.full_plydata['vertex']['label'][self.sample_ids[shot]]
        self.cluster_result = KMeans(
            n_clusters=shot,
            init=self.embedding_mat[selected_vertex_indices_in_sampled],
        ).fit(self.embedding_mat)

        selected_predicted_labels = self.cluster_result.predict(self.embedding_mat)
        selected_predicted_labels = selected_vertex_labels[selected_predicted_labels]
        selected_predicted_distances = self.cluster_result.transform(self.embedding_mat)
        self.full_predicted_labels = selected_predicted_labels[self.full2sampled]
        self.full_predicted_distances = selected_predicted_distances[self.full2sampled]
        return self

    @log(QUIET)
    def evaluate_cluster_result(self):
        assert len(self.full_predicted_labels == self.full_plydata['vertex']['label'])
        correct = (self.full_plydata['vertex']['label'] == self.full_predicted_labels).sum()
        total = len(self.full_plydata['vertex'])
        print(f'Correctness: {correct * 100/total:.2f}')
        return self

    @log(QUIET)
    def save(self):
        torch.save({
            "labels": self.full_predicted_labels,
            "confidence": self.full_predicted_distances,
        }, f"{SAVE_DIR}/{self.scan_id}_predictions.obj")
        return self

    def visualize(self):
        raise NotImplementedError 


# %%
if __name__ == '__main__':
    with count_time('calculate one scene'):
        pipeline = SpecClusterPipeline('scene0102_00')
        pipeline.downsample()
        pipeline.calc_geod_dist()
        pipeline.calc_ang_dist()
        for ratio in (0.4,):
            pipeline.calc_aff_mat(ratio)
            for feature in (100,):
                pipeline.calc_embedding(feature).knn_cluster(20).evaluate_cluster_result()
    embed()
# %%
