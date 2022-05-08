# %%
import numpy as np
from plyfile import PlyData, PlyElement

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

    # Recreate the PlyElement instance
    v = PlyElement.describe(a, 'vertex')

    # Recreate the PlyData instance
    p = PlyData([v, f], text=True)

    return p

PALETTE = [
    [82, 84, 163],
    [94, 106, 211],
    [213, 92, 176],
    [227, 119, 194],
    [96, 207, 209],
    [112, 128, 144],
    [44, 160, 44],
    [146, 111, 194],
    [178, 127, 135],
    [100, 125, 154],
    [158, 218, 229],
    [140, 153, 101],
    [143, 160, 44],
    [153, 98, 156],
    [91, 163, 138],
    [255, 127, 14],
    [172, 114, 82],
    [78, 71, 183],
    [92, 193, 61],
    [200, 54, 131],
    [51, 176, 203],
    [202, 185, 52],
    [140, 57, 197],
    [219, 219, 141],
    [66, 188, 102],
    [247, 182, 210],
    [23, 190, 207],
    [196, 156, 148],
    [148, 103, 189],
    [197, 176, 213],
    [214, 145, 40],
    [214, 39, 40],
    [255, 152, 150],
    [140, 86, 75],
    [188, 189, 34],
    [255, 187, 120],
    [31, 119, 180],
    [152, 223, 138],
    [174, 199, 232],
]

from tqdm import tqdm
def map_color_to_label(filepath, savepath):
    ply_file = PlyData.read(filepath)
    ply_file = add_fields_online(ply_file, [('label', '|u1')])
    for i in tqdm(range(len(ply_file['vertex']))):
        vertex_color = (
            ply_file['vertex'][i]['red'],
            ply_file['vertex'][i]['green'],
            ply_file['vertex'][i]['blue'],
        )
        set = False
        for idx, color in enumerate(PALETTE):
            if  color[0] == vertex_color[0] and \
                 color[1] == vertex_color[1] and \
                 color[2] == vertex_color[2]:
                 ply_file['vertex'][i]['label'] = idx
                 set = True
        if not set:
            ply_file['vertex'][i]['label'] = 255

    with open(savepath, 'wb') as f:
        ply_file.write(f)



# %%
