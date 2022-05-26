from PIL import Image

import numpy as np


CITYSCAPES_PALETTE = [(0., 0., 0.), (174., 199., 232.), (152., 223., 138.), (31., 119., 180.), (255., 187., 120.),
                      (188., 189., 34.), (140., 86., 75.), (255., 152., 150.), (214., 39., 40.), (197., 176., 213.),
                      (148., 103., 189.), (196., 156., 148.), (23., 190., 207.), (26., 13., 201.), (247., 182., 210.),
                      (66., 188., 102.), (219., 219., 141.), (140., 57., 197.), (202., 185., 52.), (51., 176., 203.),
                      (200., 54., 131.), (92., 193., 61.), (78., 71., 183.), (172., 114., 82.), (255., 127., 14.),
                      (91., 163., 138.), (153., 98., 156.), (140., 153., 101.), (158., 218., 229.), (100., 125., 154.),
                      (178., 127., 135.), (182., 56., 128.), (146., 111., 194.), (44., 160., 44.), (112., 128., 144.),
                      (96., 207., 209.), (227., 119., 194.), (213., 92., 176.), (94., 106., 211.), (82., 84., 163.),
                      (100., 85., 144.)]


def dump_points_with_labels(coords, labels, output, palette=CITYSCAPES_PALETTE):
    assert len(coords) == len(labels)
    with open(output, 'w') as f:
        for coord, label in zip(coords, labels):
            if label != 255:
                print(f"{coord[0]} {coord[1]} {coord[2]} {palette[label][0]/255} {palette[label][1]/255} {palette[label][2]/255}",
                      file=f)
            else:
                print(f"{coord[0]} {coord[1]} {coord[2]} 0.99 0.99 0.99", file=f)


def dump_points_with_features(coords, feats, output):
    assert len(coords) == len(feats)
    with open(output, 'w') as f:
        for coord, feat in zip(coords, feats):
            print(f"{coord[0]} {coord[1]} {coord[2]} {feat[0]} {feat[1]} {feat[2]}", file=f)


def get_correlated_map(ret):
    arr = ret.detach().cpu().numpy()
    return np.stack([(arr * 255).astype(np.uint8) for _ in range(3)], axis=2)

def dump_correlated_map(ret, output):
    Image.fromarray(get_correlated_map(ret)).save(output)