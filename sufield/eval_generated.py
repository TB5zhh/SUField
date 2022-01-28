# %%
import os
import numpy as np
import plyfile
from tqdm import tqdm
import torch
from IPython import embed
from config import VALID_CLASS_IDS, CLASS_LABELS

GROUND_TRUTH_DIR = '/home/aidrive/tb5zhh/3d_scene_understand/data/full/train'
GENERATED_DIR = '/home/aidrive/tb5zhh/3d_scene_understand/SUField/results/generate_datasets'
UNC_DATA_PATH = '/home/aidrive/tb5zhh/3d_scene_understand/3DScanSeg/results'
files = os.listdir('/home/aidrive/tb5zhh/3d_scene_understand/SUField/results/generate_datasets/200_fit_double/train')

GENERATE_DIR = lambda variant: f"{GENERATED_DIR}/{variant}/train"

def wrap(fn, format):
    class Proxy:
        def __init__(self, *args, **kwargs) -> None:
            self.ret = fn(*args, **kwargs)
        def __str__(self) -> str:
            return format(self.ret)
    return Proxy

def extract_labels(path):
    data = plyfile.PlyData.read(path)
    labels = data['vertex']['label']
    return np.asarray(labels)


def compute_miou(path, unc=None):
    Is = [0 for _ in range(20)]
    Os = [0 for _ in range(20)]
    valid = 0
    counting = 0
    for file in tqdm(files, desc=path):
        gt_labels = extract_labels(f"{GROUND_TRUTH_DIR}/{file}")
        if unc is None:
            eval_labels = extract_labels(f"{path}/{file}")
        else:
            unc_mapping = np.asarray(torch.load(f'{UNC_DATA_PATH}/mappings/{file.strip(".ply")}_mapping.obj')['inverse'])
            eval_labels = np.asarray(torch.load(f'{UNC_DATA_PATH}/{unc}/{file.strip(".ply")}_predicted.obj')[unc_mapping])
            eval_labels = np.asarray(VALID_CLASS_IDS)[eval_labels]

        counting += len(gt_labels)

        selector = np.bitwise_and(np.bitwise_and(gt_labels >= 0, gt_labels <= 40), np.bitwise_and(eval_labels >= 0, eval_labels <= 40))
        
        gt_labels = gt_labels[selector].astype(int)
        eval_labels = eval_labels[selector].astype(int)
        valid += len(gt_labels)

        assert gt_labels.shape == eval_labels.shape, "the label arrays have different shapes"
        hist = np.bincount(gt_labels * 41 + eval_labels, minlength=41**2).reshape(41, 41)
        Is += np.diag(hist)[np.asarray(VALID_CLASS_IDS)]
        Os += (hist.sum(0) + hist.sum(1) - np.diag(hist))[np.asarray(VALID_CLASS_IDS)]

    return [float(i / o) for i, o in zip(Is, Os)], valid / counting

format_compute_miou = lambda s: f"{s[1]*100:2f} available. Class-wise mIoU: \n" + "\n".join([f"{label:>15}: {iou*100:.2f}" for label, iou in zip(CLASS_LABELS, s[0])])

MIOU = wrap(compute_miou, format_compute_miou)


# %%



# Ground Truth
# print("200_fit_sdouble", MIOU(GENERATE_DIR("200_fit_sdouble")))
# print("200_fit_double", MIOU(GENERATE_DIR("200_fit_double")))
# print("200_fit_unc", MIOU(GENERATE_DIR("200_fit_unc")))
# print("200_fit_spec", MIOU(GENERATE_DIR("200_fit_spec")))
# print("100_fit_sdouble", MIOU(GENERATE_DIR("100_fit_sdouble")))
# print("100_fit_double", MIOU(GENERATE_DIR("100_fit_double")))
# print("100_fit_unc", MIOU(GENERATE_DIR("100_fit_unc")))
# print("100_fit_spec", MIOU(GENERATE_DIR("100_fit_spec")))
# print("50_fit_sdouble", MIOU(GENERATE_DIR("50_fit_sdouble")))
# print("50_fit_double", MIOU(GENERATE_DIR("50_fit_double")))
# print("50_fit_unc", MIOU(GENERATE_DIR("50_fit_unc")))
# print("50_fit_spec", MIOU(GENERATE_DIR("50_fit_spec")))
# print("20_fit_sdouble", MIOU(GENERATE_DIR("20_fit_sdouble")))
# print("200_fit_double", MIOU(GENERATE_DIR("200_fit_double")))
# print("200_fit_unc", MIOU(GENERATE_DIR("200_fit_unc")))
# print("200_fit_spec", MIOU(GENERATE_DIR("200_fit_spec")))
print("200_init", MIOU(GENERATE_DIR("200_fit_spec"), unc=200))
print("100_init", MIOU(GENERATE_DIR("200_fit_spec"), unc=100))
print("50_init", MIOU(GENERATE_DIR("200_fit_spec"), unc=50))
print("20_init", MIOU(GENERATE_DIR("200_fit_spec"), unc=20))

# %%

