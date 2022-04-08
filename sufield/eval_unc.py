# %%
"""This file evaluates the distribution of initial pseudo labels"""
from tqdm import tqdm
import numpy as np
import torch
from plyfile import PlyData
try:
    from .config import (CLASS_LABELS, CONF_FILE, SCANNET_COLOR_MAP, TRAIN_IDS, VALID_CLASS_IDS)
except:
    from config import (CLASS_LABELS, CONF_FILE, SCANNET_COLOR_MAP, TRAIN_IDS, VALID_CLASS_IDS)

UNC_DATA_PATH = '/home/aidrive/tb5zhh/3d_scene_understand/3DScanSeg/results'
NUM_CLS = 20
ORIGIN_PATH = '/home/aidrive/tb5zhh/3d_scene_understand/data/full_mesh/train'
num_point = 20

good_unc_distances = [[] for _ in range(NUM_CLS)]
bad1_unc_distances = [[] for _ in range(NUM_CLS)]
bad2_unc_distances = [[] for _ in range(NUM_CLS)]
for ply_name in tqdm(TRAIN_IDS, desc='load'):

    gt_plydata = PlyData.read(f"{ORIGIN_PATH}/{ply_name}.ply")
    gt_labels = gt_plydata['vertex']['label']

    unc_obj = np.asarray(torch.load(f'{UNC_DATA_PATH}/{num_point}/{ply_name}_unc.obj'))
    unc_mapping = np.asarray(torch.load(f'{UNC_DATA_PATH}/mappings/{ply_name}_mapping.obj')['inverse'])
    unc_labels = np.asarray(torch.load(f"{UNC_DATA_PATH}/{num_point}/{ply_name}_predicted.obj")[unc_mapping])
    unc_confidence = unc_obj[unc_mapping]

    for idx, cls_id in enumerate(VALID_CLASS_IDS):
        # spec_selector = np.where(spec_labels == cls_id)[0]
        good_selector = np.bitwise_and(unc_labels == idx, gt_labels == cls_id)  # ! use idx on purpose
        bad1_selector = np.bitwise_and(unc_labels == idx, gt_labels != cls_id)  # ! use idx on purpose
        bad2_selector = np.bitwise_and(unc_labels != idx, gt_labels == cls_id)  # ! use idx on purpose

        good_unc_distances[idx].append(unc_confidence[good_selector][:, idx].flatten())
        bad1_unc_distances[idx].append(unc_confidence[bad1_selector][:, idx].flatten())
        bad2_unc_distances[idx].append(unc_confidence[bad2_selector][:, idx].flatten())

        # records[idx][0] = np.hstack((records[idx][0], spec_conf))
        # records[idx][1] = np.hstack((records[idx][1], unc_conf))

from .tools import render_fit

good = []
bad = []
for i in range(NUM_CLS):
    good.append(good_unc_distances[i])
    bad.append(bad1_unc_distances[i])
    bad.append(bad2_unc_distances[i])

good = torch.as_tensor(np.hstack(good))
bad = torch.as_tensor(np.hstack(bad))
render_fit(good, [], bins=500)
render_fit(bad, [], bins=500)

# good_unc_stat = [torch.as_tensor(np.hstack(good_unc_distances[i])) for i in range(NUM_CLS)]
# bad1_unc_stat = [torch.as_tensor(np.hstack(bad1_unc_distances[i])) for i in range(NUM_CLS)]
# bad2_unc_stat = [torch.as_tensor(np.hstack(bad2_unc_distances[i])) for i in range(NUM_CLS)]
# for idx in range(20):
#     l = [
#         (good_unc_stat[idx], 'good'),
#         (torch.hstack((bad1_unc_stat[idx], bad2_unc_stat[idx])), 'bad'),
#         (bad1_unc_stat[idx], 'bad1'),
#         (bad2_unc_stat[idx], 'bad2'),
#     ]
#     for item, name in l:
#         print(idx, name)
#         try:
#             render_fit(item, [], bins=500, save=f'{idx}-{name}.png')
#             print(f'amount: {len(item)}')
#             print(f'mean: {item.mean().item()}')
#             print(f'var: {item.var().item()}')
#             print(f'std: {item.std().item()}')
#             print(f'median: {item.median().item()}')
#             print(f'maximum: {item.max().item()}')
#             print(f'minimum: {item.min().item()}')
#         except:
#             print('warning: error')
#             pass
# %%
