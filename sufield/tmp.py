
from plyfile import PlyData

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

FULL = '/home/aidrive/tb5zhh/3d_scene_understand/data/full/train/scene0006_01.ply'
FULL_MESH = '/home/aidrive/tb5zhh/3d_scene_understand/data/full_mesh/train/scene0006_01.ply'
POINT_SUPERVISION_MESH = '/home/aidrive/tb5zhh/3d_scene_understand/data/50/train/scene0006_01.ply'
FIT_SUPERVISION_MESH = '/home/aidrive/tb5zhh/3d_scene_understand/data/50_fit_double/train/scene0006_01.ply'

full = PlyData.read(FULL)
full_mesh = PlyData.read(FULL_MESH)
point_mesh = PlyData.read(POINT_SUPERVISION_MESH)
fit_mesh = PlyData.read(FIT_SUPERVISION_MESH)

point_mesh_label = point_mesh['vertex']['label']
for i in range(len(full_mesh['vertex'])):
    label = point_mesh_label[i]
    if label != 255:
        full_mesh['vertex']['red'][i] = SCANNET_COLOR_MAP[label][0]
        full_mesh['vertex']['green'][i] = SCANNET_COLOR_MAP[label][1]
        full_mesh['vertex']['blue'][i] = SCANNET_COLOR_MAP[label][2]
    else:
        full_mesh['vertex']['red'][i] = 127
        full_mesh['vertex']['green'][i] = 127
        full_mesh['vertex']['blue'][i] = 127

full_mesh.write('/home/aidrive/tb5zhh/tmp/1.ply')

point_mesh_label = fit_mesh['vertex']['label']
for i in range(len(full_mesh['vertex'])):
    label = point_mesh_label[i]
    if label != 255:
        full_mesh['vertex']['red'][i] = SCANNET_COLOR_MAP[label][0]
        full_mesh['vertex']['green'][i] = SCANNET_COLOR_MAP[label][1]
        full_mesh['vertex']['blue'][i] = SCANNET_COLOR_MAP[label][2]
    else:
        full_mesh['vertex']['red'][i] = 127
        full_mesh['vertex']['green'][i] = 127
        full_mesh['vertex']['blue'][i] = 127

full_mesh.write('/home/aidrive/tb5zhh/tmp/2.ply')

point_mesh_label = fit_mesh['vertex']['label']
for i in range(len(full_mesh['vertex'])):
    if label != 255:
        full_mesh['vertex']['red'][i] = full['vertex']['red'][i]
        full_mesh['vertex']['green'][i] = full['vertex']['green'][i]
        full_mesh['vertex']['blue'][i] = full['vertex']['blue'][i]
    else:
        full_mesh['vertex']['red'][i] = 127
        full_mesh['vertex']['green'][i] = 127
        full_mesh['vertex']['blue'][i] = 127

full_mesh.write('/home/aidrive/tb5zhh/tmp/3.ply')

import numpy as np
from IPython import embed
import torch
unc_obj = torch.load(f'/home/aidrive/tb5zhh/3d_scene_understand/3DScanSeg/results/50/scene0006_01_unc.obj')
unc_mapping = np.asarray(torch.load(f'/home/aidrive/tb5zhh/3d_scene_understand/3DScanSeg/results/mappings/scene0006_01_mapping.obj')['inverse'])
unc_labels = torch.load(f"/home/aidrive/tb5zhh/3d_scene_understand/3DScanSeg/results/50/scene0006_01_predicted.obj")[unc_mapping]
unc_confidence = unc_obj[unc_mapping]
unc_confidence = unc_confidence.to(torch.float64)
unc_confidence = torch.gather(unc_confidence, 1, unc_labels.reshape(1, -1))
unc_confidence = unc_confidence.flatten()
unc_confidence = unc_confidence / unc_confidence.max()
for i in range(len(full_mesh['vertex'])):
    ratio = unc_confidence[i]
    # embed()
    full_mesh['vertex']['red'][i] = 127 + ratio * 128
    full_mesh['vertex']['green'][i] = 127
    full_mesh['vertex']['blue'][i] = 127 + (1 - ratio) * 128
full_mesh.write('/home/aidrive/tb5zhh/tmp/4.ply')

spec_obj = torch.load('/home/aidrive/tb5zhh/3d_scene_understand/SUField/results/spec_predictions/scene0006_01_50.obj')
spec_confidence = torch.as_tensor(spec_obj['confidence'])
spec_confidence = spec_confidence / spec_confidence.max()
for i in range(len(full_mesh['vertex'])):
    ratio = spec_confidence[i]
    full_mesh['vertex']['red'][i] = 127 + i * 128
    full_mesh['vertex']['green'][i] = 127 + (1 - i) * 128
    full_mesh['vertex']['blue'][i] = 127 
full_mesh.write('/home/aidrive/tb5zhh/tmp/5.ply')
