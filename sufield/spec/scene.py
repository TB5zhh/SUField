# %%
import argparse
import time
import sys
from functools import wraps
from multiprocessing import Pool, shared_memory
import os
import sys
from datetime import datetime
from typing import Tuple

import numpy as np
import open3d as o3d
import potpourri3d as pp3d
from potpourri3d import point_cloud
import pymeshlab
import torch
from IPython import embed
from plyfile import PlyData, PlyElement
from sklearnex import patch_sklearn
from tqdm import tqdm
import configparser
from pathos.pools import ProcessPool

patch_sklearn()
from sklearn.cluster import KMeans
from sufield.config import CONF_FILE
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


def plydata_to_arrays(plydata: PlyData) -> Tuple[np.ndarray, np.ndarray]:
    vertices = np.vstack((plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z'])).T
    faces = np.stack(plydata['face']['vertex_indices'])
    return vertices, faces


# %%


class SpecClusterPipeline():

    def __init__(self, scan_id: str) -> None:
        self.scan_id = scan_id
        self.conf = configparser.ConfigParser()
        self.conf.read(CONF_FILE)
        self.conf = self.conf['SpectralClustering']

    def downsample(self, plydata: PlyData) -> PlyData:

        # TODO restore color of the meshes
        temp_ply_path = f'.tmp_{self.scan_id}.ply'

        plydata.write(temp_ply_path)
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

        return new_ply_data

    def calc_geod_dist(self, plydata):
        vertices, faces = plydata_to_arrays(plydata)
        solver = pp3d.MeshHeatMethodDistanceSolver(vertices, faces)
        distances = []
        for i in tqdm(range(len(plydata['vertex']))):
            distances.append(solver.compute_distance(i))
        geod_mat = np.stack(distances)
        geod_mat = (geod_mat + geod_mat.T) / 2
        return geod_mat

    def calc_geod_dist_parallel(self, plydata: PlyData) -> np.ndarray:
        start = datetime.now()
        # TODO accelerate
        point_count = len(plydata['vertex'])
        vertices, faces = plydata_to_arrays(plydata)

        ######### PARALLEL GENERATION ########
        target = np.ndarray((vertices.shape[0], vertices.shape[0]), dtype=float)

        shm_a = shared_memory.SharedMemory(create=True, size=vertices.nbytes)
        shared_vertices = np.ndarray(vertices.shape, dtype=vertices.dtype, buffer=shm_a.buf)
        shm_b = shared_memory.SharedMemory(create=True, size=faces.nbytes)
        shared_faces = np.ndarray(faces.shape, dtype=faces.dtype, buffer=shm_b.buf)
        shm_target = shared_memory.SharedMemory(create=True, size=target.nbytes)
        shared_target = np.ndarray(target.shape, dtype=target.dtype, buffer=shm_target.buf)

        shared_vertices[:] = vertices[:]
        shared_faces[:] = faces[:]

        def run(args):
            start, end, v_name, f_name, t_name = args
            shm_a = shared_memory.SharedMemory(name=v_name)
            shm_b = shared_memory.SharedMemory(name=f_name)
            shm_target = shared_memory.SharedMemory(name=t_name)
            shared_vertices = np.ndarray(vertices.shape, dtype=vertices.dtype, buffer=shm_a.buf)
            shared_faces = np.ndarray(faces.shape, dtype=faces.dtype, buffer=shm_b.buf)
            shared_target = np.ndarray(target.shape, dtype=target.dtype, buffer=shm_target.buf)
            solver = pp3d.MeshHeatMethodDistanceSolver(shared_vertices, shared_faces, t_coef=self.conf.getfloat('MeshHeatT'))
            for idx in range(start, end):
                shared_target[idx] = solver.compute_distance(idx)

        nproc = 4
        pool = ProcessPool(nodes=nproc)
        results = pool.amap(run, [(idx, min(idx, idx + point_count // nproc), shm_a.name, shm_b.name, shm_target.name) for idx in range(nproc)])
        while not results.ready():
            print(".", end='', file=sys.stderr)
            time.sleep(1)
        pool.close()
        pool.join()

        geod_mat = np.copy(shared_target)
        geod_mat = (geod_mat + geod_mat.T) / 2
        # TODO check normalization method
        geod_mat = geod_mat / geod_mat.mean()
        shm_a.close()
        shm_a.unlink()
        shm_b.close()
        shm_b.unlink()
        shm_target.close()
        shm_target.unlink()

        end = datetime.now()
        print(f'spent: {(end - start).seconds}')
        return geod_mat

    def calc_ang_dist(self, plydata) -> np.ndarray:
        pcd = o3d.geometry.PointCloud()
        vertices = plydata_to_arrays(plydata)
        pcd.points = o3d.utility.Vector3dVector(vertices)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=self.conf.getint('NormalKnnRange')))
        center = vertices.mean(dim=0)
        pcd.orient_normals_towards_camera_location(center)
        normals = np.asarray(pcd.normals)

        # use absolute result
        t = 1 - (normals @ normals.T).absolute()
        ang_mat = t / t.mean()
        return ang_mat

    def calc_aff_mat(self, _geod_mat, _ang_mat) -> np.ndarray:
        geod_mat = torch.tensor(_geod_mat).cuda()
        ang_mat = torch.tensor(_ang_mat).cuda()
        ratio = self.conf.getfloat('DistanceProportion')
        dist_mat = ratio * geod_mat + (1 - ratio) * ang_mat
        dist_mat = (dist_mat + dist_mat.T) / 2
        sigma = dist_mat.mean() if self.conf.getboolean('AutoTemperature') else self.conf.getfloat('Temperature')
        aff_mat = np.e**(-dist_mat / (2 * sigma**2))

        aff_iv_mat = torch.diag(1 / aff_mat.sum(dim=1).sqrt())
        n_mat = (aff_iv_mat @ aff_mat @ aff_iv_mat)
        return np.array(n_mat.cpu())

    def calc_embedding(self, _aff_mat) -> np.ndarray:
        aff_mat = torch.tensor(_aff_mat).cuda()
        eigh_vals, eigh_vecs = torch.linalg.eigh(aff_mat.cuda())
        eigh_vecs = eigh_vecs.T
        embedding = eigh_vecs[-self.conf.getint('Shots'):, :].flip(dims=(0,)).T
        embedding_cpu = embedding.cpu()
        return np.array(embedding_cpu)

    def knn_cluster(self, embed_mat):
        raise NotImplementedError

    def upsample(self,):
        raise NotImplementedError


# %%
if __name__ == '__main__':
    pipeline = SpecClusterPipeline('a')
    pd = PlyData.read('/home/aidrive/tb5zhh/SUField/data/scannetv2/scans/scene0702_00/scene0702_00_vh_clean_2.labels.ply')
    pd = pipeline.downsample(pd)
    print('start')
    # mat = pipeline.calc_geod_dist_naive(pd)
    # np.save('0.npy', mat)

    mat = pipeline.calc_geod_dist(pd)
    np.save('1.npy', mat)
# %%
