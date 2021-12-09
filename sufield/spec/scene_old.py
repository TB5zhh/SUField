# %%
"""
This file provides methods to preprocess a complete scene to a sampled, few-shot labeled one
"""

# %%

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import open3d as o3d
import potpourri3d as pp3d
import pymeshlab as ml
import torch
from IPython import embed
from plyfile import PlyData, PlyElement
from sklearnex import patch_sklearn
from tqdm import tqdm

patch_sklearn()
from sklearn.cluster import KMeans
from sufield.config import *

# %%

# Definition:
# ply_name: scene0000_00
# ply_file: scene0000_00.ply
# ply_path: ../../../scene0000_00.ply


def dye_ratio(plydata, output, mat, idx, prop=10, zoom=False):
    """
    Apply color to a 3D scene according to the distance between points, and save as a ply file.
    The idx point is dyed white, whereas other points are dyed according to their distances towards idx point.
    
    only used in run(ply_path, args)
    """
    for i in range(len(mat[idx])):
        if zoom:
            ratio = (mat[idx][i] - mat[idx].min()) / (mat[idx].max() - mat[idx].min()) * prop
        else:
            ratio = mat[idx][i] / mat[idx].max() * prop
        plydata['vertex']['red'][i] = int(ratio * 255)
        plydata['vertex']['green'][i] = int((1 - ratio) * 255)
        plydata['vertex']['blue'][i] = int(ratio * 255) / 2
    plydata['vertex']['red'][idx] = 255
    plydata['vertex']['green'][idx] = 255
    plydata['vertex']['blue'][idx] = 255
    with open(output, mode='wb') as f:
        plydata.write(f)


def dye_normal(plydata, normals, output):
    """
    Apply color to a 3D scene according to the normal vector of each point, and save as a ply file.

    only used in run(ply_path, args)
    The normal vector, often simply called the "normal," to a surface is a vector which is perpendicular to the surface at a given point.
    """
    for i in range(len(normals)):
        plydata['vertex']['red'][i] = int((normals[i][0]) * 127 + 127)
        plydata['vertex']['green'][i] = int((normals[i][1]) * 127 + 127)
        plydata['vertex']['blue'][i] = int((normals[i][2]) * 127 + 127)
    with open(output, mode='wb') as f:
        plydata.write(f)


def dye_class(plydata, labels, confidence, output):
    """
    Apply color to a 3D scene according to labels and save as a ply file.
    """
    for i in range(len(plydata['vertex'])):
        plydata['vertex']['red'][i],\
        plydata['vertex']['green'][i],\
        plydata['vertex']['blue'][i]\
        = SCANNET_COLOR_MAP.get(labels[i], (255, 0, 0)) if confidence[i] < 0.4 else (255, 255, 255)
        # self.plydata['vertex']['nx'][i],\
        # self.plydata['vertex']['ny'][i],\
        # self.plydata['vertex']['nz'][i]\
        # = self.V_normal[i]
    with open(output, mode='wb') as f:
        plydata.write(f)


# %%
def downsample(file_path: str, result_name: str, times: int):
    """
    Use tools from MeshLab to downsample a scene for specified times. from file to file

    The downsample operation perserves colors.

    After downsampling a scene, the `label` field in the scene will be lost,
    and should be restored manually.

    Due to bad implement in MeshLab, working directory should be changed. 
    
    TODO (george): avoid using absolute paths 
    """
    bak = os.getcwd()
    ms = ml.MeshSet()

    os.chdir('/'.join(file_path.split('/')[:-1]))
    ms.load_new_mesh(file_path.split('/')[-1])
    os.chdir(bak)

    for _ in range(times):
        ms.apply_filter('simplification_quadric_edge_collapse_decimation')
    ms.apply_filter('remove_unreferenced_vertices')

    os.chdir('/'.join(result_name.split('/')[:-1]))
    ms.save_current_mesh(result_name.split('/')[-1], save_textures=True)
    os.chdir(bak)


def add_fields(filename, output, add_fields=[
    ('nx', 'double'),
    ('ny', 'double'),
    ('nz', 'double'),
]):
    """
    Create new PlyData structure according to input plydata with added fields, and save as new ply files
    """
    p = PlyData.read(filename)
    v, f = p.elements
    a = np.empty(len(v.data), v.data.dtype.descr + add_fields)
    for name in v.data.dtype.fields:
        a[name] = v[name]
    # a[''] = 0
    # Recreate the PlyElement instance
    v = PlyElement.describe(a, 'vertex')

    # Recreate the PlyData instance
    p = PlyData([v, f], text=True)

    p.write(output)


def add_fields_online(plydata, add_fields=[
    ('nx', 'double'),
    ('ny', 'double'),
    ('nz', 'double'),
]):
    """
    Add fields in the PlyData structure ONLINE
    """
    p = plydata
    v, f = p.elements
    a = np.empty(len(v.data), v.data.dtype.descr + add_fields)
    for name in v.data.dtype.fields:
        a[name] = v[name]
    # a[''] = 0
    # Recreate the PlyElement instance
    v = PlyElement.describe(a, 'vertex')

    # Recreate the PlyData instance
    p = PlyData([v, f], text=True)

    return p


def extract_labels(filename):
    """
    Extract labels from a local ply file.

    Return a dictionary where keys are hashed coordinates and values are labels.
    """
    with open(filename, mode='rb') as f:
        plydata = PlyData.read(f)
    d = {}
    for i in range(len(plydata['vertex'])):
        index = '$'.join((
            f"{plydata['vertex']['red'][i]}",
            f"{plydata['vertex']['green'][i]}",
            f"{plydata['vertex']['blue'][i]}",
        ))
        d[index] = plydata['vertex']['label'][i]
    return d


def apply_labels(d, filename, output):
    """NOT USED"""
    add_fields(filename, filename, [('label', 'ushort')])
    with open(filename, mode='rb') as f:
        plydata = PlyData.read(f)

    for i in range(len(plydata['vertex'])):
        index = '$'.join((
            f"{plydata['vertex']['red'][i]}",
            f"{plydata['vertex']['green'][i]}",
            f"{plydata['vertex']['blue'][i]}",
        ))
        plydata['vertex']['label'][i] = d[index]
    plydata.write(output)


def test_labels(filename, output, t=0):
    """NOT USED"""
    with open(filename, mode='rb') as f:
        plydata = PlyData.read(f)
    for i in range(len(plydata['vertex'])):
        plydata['vertex']['red'][i],\
        plydata['vertex']['green'][i],\
        plydata['vertex']['blue'][i]\
        = (255,0,0) if plydata['vertex']['label'][i] == 0 else (255,255,255)
    with open(output, mode='wb') as f:
        plydata.write(f)


def scene_name(file_name: str):
    """
    Return scene names according to file names

    'scene0000_02_vh_clean_2.labels.ply' -> 'scene0000_02'
    """
    # return '_'.join(file_name.split('_')[:2])
    return file_name.split('.')[0]


# %%
def run(ply_path, args):
    """
    Processing pipeline

    - `ply_path`: full path to a ply file
    - `args`: other arguments
    """
    totalstarttime = datetime.now()
    starttime = datetime.now()

    # ply file name
    # Only process scenes which has few-shot labels
    ply_file = ply_path.split('/')[-1]
    ply_name = scene_name(ply_file)
    if ply_name not in torch.load(f'indices/{args.shots}.dict'):
        return
    if os.path.isfile(f'new_results_{args.shots}/data/{ply_name}-ds_prediction-{args.dist_proportion}.obj'):
        tmp = torch.load(f'new_results_{args.shots}/data/{ply_name}-ds_prediction-{args.dist_proportion}.obj')
        if 'centers' in tmp.keys():
            return
    print(f"Processing {ply_path} ...")

    try:
        plydata = PlyData.read(f'new_results_{args.shots}/sampled/{ply_name}-full_labeled.ply')
        ds_labels = plydata['vertex']['label']
        plydata = PlyData.read(f'new_results_{args.shots}/sampled/{ply_name}-fewshot_labeled-colored.ply')
        tmp = torch.load(f'new_results_{args.shots}/data/{ply_name}-ds_mapping.obj')
        ds_indices2full_indices = tmp['mapping']
        full_indices2ds_indices = tmp['inverse']
        ds_coords = torch.tensor(
            np.stack((plydata['vertex']['x'], \
                        plydata['vertex']['y'], \
                        plydata['vertex']['z']),
                        axis=1))
        
        selected_label = torch.load(f'new_results_{args.shots}/data/{ply_name}-selected.obj')['labels']

        ds_selected_indices = torch.load(f'new_results_{args.shots}/data/{ply_name}-ds_selected_indices.obj')
    except Exception as e:
        print(e)
        # The indices of points that should have labels (selected points) in the scene
        selected_indices = torch.tensor( \
            torch.load(f'indices/{args.shots}.dict')[ply_name])
        full_plydata = PlyData.read(f'{ply_path}')
        full_labels = torch.tensor(full_plydata['vertex']['label'].astype(np.int32))
        full_coords = torch.tensor( \
            np.stack((
                full_plydata['vertex']['x'],
                full_plydata['vertex']['y'],
                full_plydata['vertex']['z'],
            ), axis=1))

        # Filter the coordinates and labels of selected points
        selected_coords = full_coords.index_select(0, selected_indices)
        selected_label = full_labels.index_select(0, selected_indices)
        torch.save({
            'labels': selected_label
        }, f'new_results_{args.shots}/data/{ply_name}-selected.obj')

        # + ======================
        # Downsample the scene
        size = len(full_coords)
        times = 0
        while size > 20000:
            times += 1
            size //= 2
        downsample(ply_path, f'new_results_{args.shots}/sampled/{ply_name}-sampled.ply', times)

        # Read downsampled scene and add required fields
        plydata = PlyData.read(f'new_results_{args.shots}/sampled/{ply_name}-sampled.ply')
        plydata = add_fields_online(plydata, [('label', 'ushort')])
        plydata = add_fields_online(plydata, [('nx', 'float'), ('ny', 'float'), ('nz', 'float')])

        # Coords of all points in the downsampled scene
        ds_coords = torch.tensor(
            np.stack((plydata['vertex']['x'], \
                        plydata['vertex']['y'], \
                        plydata['vertex']['z']),
                        axis=1))

        ds_coords = ds_coords.cuda()
        full_coords = full_coords.cuda()
        # For each point in downsampled scene
        # Calculate the index of the closest point in the origin scene
        ds_indices2full_indices = torch.tensor([((full_coords - testant)**2).sum(dim=1).min(dim=0)[1] for testant in ds_coords])

        # For mapping results inversely
        full_indices2ds_indices = torch.tensor([((testant - ds_coords)**2).sum(dim=1).min(dim=0)[1] for testant in full_coords])
        ds_coords = ds_coords.cpu()
        full_coords = full_coords.cpu()

        torch.save({
            'mapping': ds_indices2full_indices.cpu(),
            'inverse': full_indices2ds_indices.cpu(),
        }, f'new_results_{args.shots}/data/{ply_name}-ds_mapping.obj')

        # Select the labels of those points in the original scene
        # and treat as the labels of points in the downsampled scene
        ds_labels = full_labels.index_select(0, ds_indices2full_indices)

        # Apply labels to all points in downsampled plydata, and save
        plydata['vertex']['label'] = np.array(ds_labels.cpu())
        plydata.write(f'new_results_{args.shots}/sampled/{ply_name}-full_labeled.ply')

        # For each labeled point in original scene
        # Calcualte the index of the closest point in the downsampled scene
        # This traverse preserves the order of `selected_coords`
        ds_selected_indices = torch.tensor([((ds_coords - target)**2).sum(dim=1).min(dim=0)[1] for target in selected_coords])
        torch.save(ds_selected_indices, f'new_results_{args.shots}/data/{ply_name}-ds_selected_indices.obj')
        # selected_coords, ds_selected_indices, selected_labels are in order
        # Construct mapper from point indices to labels
        ds_indices2labels = {idx.item(): int(label.item()) for idx, label in zip(ds_selected_indices, selected_label)}

        # embed()

        # Clear labels of unselected points, and apply color to selected points
        plydata['vertex']['label'] = np.array([ds_indices2labels.get(i, 255) for i in range(len(plydata['vertex']))])
        for i in range(len(plydata['vertex'])):
            plydata['vertex']['red'][i],\
            plydata['vertex']['green'][i],\
            plydata['vertex']['blue'][i]\
            = SCANNET_COLOR_MAP.get(plydata['vertex']['label'][i], (255, 0, 0)) \
                if plydata['vertex']['label'][i] != 255 else (255, 255, 255)
        plydata.write(f'new_results_{args.shots}/sampled/{ply_name}-fewshot_labeled-colored.ply')
    print("Preprocess done.")

    endtime = datetime.now()
    print(f'\tspent: {(endtime-starttime).seconds} s')
    # + ======================
    starttime = datetime.now()

    ######## ! Calculation starts ############

    # Geodesic distances
    # ! Cache too large, don't save this object anymore
    geod_cache_path = f'new_results_{args.shots}/data/{ply_name}-ds_geod.obj'
    try:
        geod_mat = torch.load(geod_cache_path)
    except Exception as e:
        distances = []
        gd_solver = pp3d.MeshHeatMethodDistanceSolver(*pp3d.read_mesh(f'new_results_{args.shots}/sampled/{ply_name}-full_labeled.ply'))
        for i in range(len(plydata['vertex'])):
            distances.append(gd_solver.compute_distance(i))
        with torch.no_grad():
            t = torch.tensor(distances)
            geod_mat = t / t.mean()
        # torch.save(geod_mat, geod_cache_path)
    print("Geod mat calc done.")

    # embed()

    endtime = datetime.now()
    print(f'\tspent: {(endtime-starttime).seconds} s')
    # + ======================
    starttime = datetime.now()

    # Angular distances
    # The normals of points are calculated here
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ds_coords.cpu())
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=args.normal_knn_range))
    # pcd.orient_normals_to_align_with_direction([0, 0, 1])
    # pcd.orient_normals_consistent_tangent_plane(k=args.normal_align_knn_range)
    ds_coords_center = ds_coords.mean(dim=0)
    pcd.orient_normals_towards_camera_location(ds_coords_center.numpy())

    with torch.no_grad():
        ds_normals = torch.tensor(np.asarray(pcd.normals))
        plydata['vertex']['nx'] = ds_normals[:, 0]
        plydata['vertex']['ny'] = ds_normals[:, 1]
        plydata['vertex']['nz'] = ds_normals[:, 2]
        t = 1 - ds_normals @ ds_normals.T
        ang_mat = t / t.mean()
    print("Ang mat calc done.")

    endtime = datetime.now()
    print(f'\tspent: {(endtime-starttime).seconds} s')
    # + ======================
    starttime = datetime.now()

    # Save some visualized results
    plydata.write(f'new_results_{args.shots}/sampled/{ply_name}-fewshot_labeled-normalized.ply')
    dye_ratio(plydata, f'new_results_{args.shots}/visualized/{ply_name}-geod.ply', geod_mat, 100, 1, True)
    dye_ratio(plydata, f'new_results_{args.shots}/visualized/{ply_name}-ang.ply', ang_mat, 100, 1, True)
    dye_normal(plydata, ds_normals, f'new_results_{args.shots}/visualized/{ply_name}-normal.ply')

    try:
        embedding_cpu = torch.load(f'new_results_{args.shots}/data/{ply_name}-ds_embedding-{args.dist_proportion}.obj')
    except:
        with torch.no_grad():
            # Affinity matrix
            geod_mat = geod_mat.cuda()
            ang_mat = ang_mat.cuda()
            dist_mat = args.dist_proportion * geod_mat + (1 - args.dist_proportion) * ang_mat
            dist_mat = (dist_mat + dist_mat.T) / 2
            if args.auto_exp_temp:
                sigma = dist_mat.mean()
            else:
                sigma = args.exp_temp
            aff_mat = np.e**(-dist_mat / (2 * sigma**2))  # Exponential kernel
            pdeg_mat = torch.diag(1 / aff_mat.sum(dim=1).sqrt())  # 'D'^(-1/2)
            n_mat = (pdeg_mat @ aff_mat @ pdeg_mat)
            print(f"Aff mat calc done with exp temperature {sigma:.4f}, proportion p={args.dist_proportion:.4f}.")

            endtime = datetime.now()
            print(f'\tspent: {(endtime-starttime).seconds} s')
            # + ======================
            starttime = datetime.now()

            # Embeddings of points calculated from eigenvectors and eigenvalues
            eigh_vals, eigh_vecs = torch.linalg.eigh(n_mat.cuda())
            eigh_vecs = eigh_vecs.T
            embedding = eigh_vecs[-args.shots:, :].flip(dims=(0,)).T
            embedding = \
                embedding / embedding.norm(dim=1,keepdim=True)
            embedding_cpu = embedding.cpu()
            torch.save(embedding_cpu, f'new_results_{args.shots}/data/{ply_name}-ds_embedding-{args.dist_proportion}.obj')

    print("Embedding mat calc done.")
    endtime = datetime.now()
    print(f'\tspent: {(endtime-starttime).seconds} s')
    # + ======================
    starttime = datetime.now()

    # Use selected indices for clustering initialization
    ds_normals = ds_normals.cpu()
    ds_selected_indices = ds_selected_indices.cpu()
    # embed()
    cl_result = KMeans(n_clusters=args.shots, init=embedding_cpu.index_select(0, ds_selected_indices)).fit(embedding_cpu)
    # Transfrom indices of selected points to indices of labels
    label_indices = list(int(selected_label[idx].item()) if idx >= 0 and idx < args.shots else 128 for idx in cl_result.labels_)
    # Obtain confidence of the predicted labels
    confidence = cl_result.transform(embedding_cpu).min(axis=1)
    centers = cl_result.cluster_centers_

    torch.save({
        'labels': label_indices,
        'confidence': confidence,
        'centers': centers,
    }, f'new_results_{args.shots}/data/{ply_name}-ds_prediction-{args.dist_proportion}.obj')

    # + ======================

    predicted = 0
    correct = 0
    for i in range(len(plydata['vertex'])):
        if confidence[i] < args.confidence:
            predicted += 1
            correct += (label_indices[i] == ds_labels[i])
            plydata['vertex']['label'][i] = label_indices[i]
            plydata['vertex']['red'][i],\
            plydata['vertex']['green'][i],\
            plydata['vertex']['blue'][i] = SCANNET_COLOR_MAP.get(label_indices[i], (255, 0, 0))
        else:
            plydata['vertex']['label'][i] = 255
            plydata['vertex']['red'][i],\
            plydata['vertex']['green'][i],\
            plydata['vertex']['blue'][i] = (255, 255, 255)
    shortname = '_'.join(ply_file.split('_')[:2])
    print(f"{shortname} predicted {correct} correctly out of {predicted} predicted in {len(plydata['vertex'])} vertices, " +
          f"predict rate: {predicted / (len(plydata['vertex'])):.4f}, accuracy: {correct / predicted:.4f}")
    plydata.write(f'new_results_{args.shots}/clustered/{ply_name}-clustered-{args.dist_proportion}.ply')
    endtime = datetime.now()
    print(f'\tspent: {(endtime-starttime).seconds} s')
    # with open('finished.out', mode='a') as f:
    #     f.write(f'{ply_name}\n')

    # Restoring
    starttime = datetime.now()

    FULL_DATA_PATH = '/home/aidrive/tb5zhh/data/full_mesh/train'    
    ply_data = PlyData.read(f'{FULL_DATA_PATH}/{ply_name}.ply')
    ds_predictions = label_indices
    ds_mapping = full_indices2ds_indices
    for i in range(len(ply_data['vertex'])):
        ply_data['vertex']['label'][i] = ds_predictions[ds_mapping[i]]
        ply_data['vertex']['red'][i],\
        ply_data['vertex']['green'][i],\
        ply_data['vertex']['blue'][i] = SCANNET_COLOR_MAP.get(ds_predictions[ds_mapping[i]], (255, 0, 0))
    ply_data.write(f'new_results_{args.shots}/clustered/{ply_name}-clustered-raw-{args.dist_proportion}.ply')

    ply_data_geod = PlyData.read(f'new_results_{args.shots}/visualized/{ply_name}-geod.ply')
    for i in range(len(ply_data['vertex'])):
        ply_data['vertex']['red'][i] = ply_data_geod['vertex']['red'][ds_mapping[i]]
        ply_data['vertex']['green'][i] = ply_data_geod['vertex']['green'][ds_mapping[i]]
        ply_data['vertex']['blue'][i] = ply_data_geod['vertex']['blue'][ds_mapping[i]]
    ply_data.write(f'new_results_{args.shots}/visualized/{ply_name}-geod-raw.ply')
    # restore(ply_path, args)
    ply_data = add_fields_online(ply_data, [('nx', 'float'), ('ny', 'float'), ('nz', 'float')])
    ds_coords = torch.tensor(
        np.stack((ply_data['vertex']['x'], \
                    ply_data['vertex']['y'], \
                    ply_data['vertex']['z']),
                    axis=1)) 
    ds_coords_center = ds_coords.mean(dim=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ds_coords.cpu())
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=args.normal_knn_range))
    # pcd.orient_normals_to_align_with_direction([0, 0, 1])
    # pcd.orient_normals_consistent_tangent_plane(k=args.normal_align_knn_range)
    pcd.orient_normals_towards_camera_location(ds_coords_center.numpy())
    with torch.no_grad():
        ds_normals = torch.tensor(np.asarray(pcd.normals))
        ply_data['vertex']['nx'] = ds_normals[:, 0]
        ply_data['vertex']['ny'] = ds_normals[:, 1]
        ply_data['vertex']['nz'] = ds_normals[:, 2]

    dye_normal(ply_data, ds_normals, f'new_results_{args.shots}/visualized/{ply_name}-normal-raw.ply')
    endtime = datetime.now()
    print('Restore complete')
    print(f'\tspent: {(endtime-starttime).seconds} s')
    print(f'\tTotal spent: {(endtime-totalstarttime).seconds} s')


# %%
def restore(ply_path, args):
    
    starttime = datetime.now()

    ply_name = ply_path.split('/')[-1].split('.')[0]
    ply_data = PlyData.read(f'{DATA_PATH}/{ply_name}.ply')
    ds_predictions = torch.load(f'new_results_{args.shots}/data/{ply_name}-ds_prediction-{args.dist_proportion}.obj')['labels']
    ds_mapping = torch.load(f'new_results_{args.shots}/data/{ply_name}-ds_mapping.obj')['inverse']
    for i in range(len(ply_data['vertex'])):
        ply_data['vertex']['label'][i] = ds_predictions[ds_mapping[i]]
        ply_data['vertex']['red'][i],\
        ply_data['vertex']['green'][i],\
        ply_data['vertex']['blue'][i] = SCANNET_COLOR_MAP.get(ds_predictions[ds_mapping[i]], (255, 0, 0))
    ply_data.write(f'new_results_{args.shots}/clustered/{ply_name}-clustered-raw-{args.dist_proportion}.ply')

    ply_data_geod = PlyData.read(f'new_results_{args.shots}/visualized/{ply_name}-geod.ply')
    for i in range(len(ply_data['vertex'])):
        ply_data['vertex']['red'][i] = ply_data_geod['vertex']['red'][ds_mapping[i]]
        ply_data['vertex']['green'][i] = ply_data_geod['vertex']['green'][ds_mapping[i]]
        ply_data['vertex']['blue'][i] = ply_data_geod['vertex']['blue'][ds_mapping[i]]
    ply_data.write(f'new_results_{args.shots}/visualized/{ply_name}-geod-raw.ply')
    
    # ply_data_geod = PlyData.read(f'new_results_{args.shots}/visualized/{ply_name}-normal.ply')
    # for i in range(len(ply_data['vertex'])):
    #     ply_data['vertex']['red'][i] = ply_data_geod['vertex']['red'][ds_mapping[i]]
    #     ply_data['vertex']['green'][i] = ply_data_geod['vertex']['green'][ds_mapping[i]]
    #     ply_data['vertex']['blue'][i] = ply_data_geod['vertex']['blue'][ds_mapping[i]]
    # ply_data.write(f'new_results_{args.shots}/visualized/{ply_name}-normal-raw.ply')
    ply_data = add_fields_online(ply_data, [('nx', 'float'), ('ny', 'float'), ('nz', 'float')])
    ds_coords = torch.tensor(
        np.stack((ply_data['vertex']['x'], \
                    ply_data['vertex']['y'], \
                    ply_data['vertex']['z']),
                    axis=1)) 
    ds_coords_center = ds_coords.mean(dim=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ds_coords.cpu())
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=args.normal_knn_range))
    # pcd.orient_normals_to_align_with_direction([0, 0, 1])
    # pcd.orient_normals_consistent_tangent_plane(k=args.normal_align_knn_range)
    pcd.orient_normals_towards_camera_location(ds_coords_center.numpy())

    with torch.no_grad():
        ds_normals = torch.tensor(np.asarray(pcd.normals))
        ply_data['vertex']['nx'] = ds_normals[:, 0]
        ply_data['vertex']['ny'] = ds_normals[:, 1]
        ply_data['vertex']['nz'] = ds_normals[:, 2]

    dye_normal(ply_data, ds_normals, f'new_results_{args.shots}/visualized/{ply_name}-normal-raw.ply')
    endtime = datetime.now()
    print('Restore complete')
    print(f'\tspent: {(endtime-starttime).seconds} s')


DATA_PATH = '/home/aidrive/tb5zhh/data/full_mesh/train'
if __name__ == '__main__':
    args = argparse.Namespace(**{
        'normal_knn_range': 10,
        'normal_align_knn_range': 10,
        'dist_proportion': 0.3,
        'auto_exp_temp': True,
        'exp_temp': 0.1,
        'confidence': 100,
        'shots': 42,
    })
    import sys
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    args.shots=int(sys.argv[3])
    print(args.shots)
#    iters = sorted(os.listdir(DATA_PATH))[244:750]
#    iters = sorted(os.listdir(DATA_PATH))[852:]
    iters = sorted(os.listdir(DATA_PATH))
    for id, ply_file in enumerate(iters[start:min(end, len(iters))]):
        print(f"============================= {ply_file} #{id}/{min(end,len(iters))-start} =============================")
        run(f"{DATA_PATH}/{ply_file}", args)
        torch.cuda.empty_cache()

    print('Done.')
# %%
