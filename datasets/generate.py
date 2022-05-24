import os
import sys
from tqdm import tqdm
from plyfile import PlyData
import numpy as np

def generate_bundle(scannet_path, export_dir, id=0, total=1):
    os.makedirs(export_dir, exist_ok=True)
    train_bundle = {
        'data': [],
        'feats': [],
        'label': []
    }
    total_sum = len(os.listdir(f'{scannet_path}/scans'))
    duty = sorted(os.listdir(f'{scannet_path}/scans'))[id * (total_sum  + total - 1 )// total : (id + 1) * (total_sum  + total - 1 ) // total]
    for scene_name in tqdm(duty):
        data = PlyData.read(f'{scannet_path}/scans/{scene_name}/{scene_name}_vh_clean_2.ply')
        train_bundle['data'].append(np.stack((data['vertex']['x'], data['vertex']['y'], data['vertex']['z']), axis=1))
        train_bundle['feats'].append(np.stack((data['vertex']['red'], data['vertex']['green'], data['vertex']['blue']), axis=1))

        data = PlyData.read(f'{scannet_path}/scans/{scene_name}/{scene_name}_vh_clean_2.labels.ply')
        train_bundle['label'].append(data['vertex']['label'].astype(np.uint8))

    np.save(f'{export_dir}/scannet_train_{id}.npy', train_bundle)
    
    val_bundle = {
        'data': [],
        'feats': [],
    }
    total_sum = len(os.listdir(f'{scannet_path}/scans_test'))
    duty = sorted(os.listdir(f'{scannet_path}/scans_test'))[id * (total_sum  + total - 1 )// total : (id + 1) * (total_sum  + total - 1 ) // total]
    for scene_name in tqdm(duty):
        data = PlyData.read(f'{scannet_path}/scans_test/{scene_name}/{scene_name}_vh_clean_2.ply')
        val_bundle['data'].append(np.stack((data['vertex']['x'], data['vertex']['y'], data['vertex']['z']), axis=1))
        val_bundle['feats'].append(np.stack((data['vertex']['red'], data['vertex']['green'], data['vertex']['blue']), axis=1))

    np.save(f'{export_dir}/scannet_test_{id}.npy', val_bundle)

def generate_list(scannet_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    train_data_f = open(f'{save_dir}/ScanNetv2_train_data.txt', 'w')
    train_label_f = open(f'{save_dir}/ScanNetv2_train_labels.txt', 'w')
    test_data_f = open(f'{save_dir}/ScanNetv2_test_data.txt', 'w')
    for scene_name in sorted(os.listdir(f'{scannet_path}/scans')):
        print(f'{scannet_path}/scans/{scene_name}/{scene_name}_vh_clean_2.ply', file=train_data_f)
        print(f'{scannet_path}/scans/{scene_name}/{scene_name}_vh_clean_2.labels.ply', file=train_label_f)
    for scene_name in sorted(os.listdir(f'{scannet_path}/scans_test')):
        print(f'{scannet_path}/scans_test/{scene_name}/{scene_name}_vh_clean_2.ply', file=test_data_f)
    train_data_f.close()
    train_label_f.close()
    test_data_f.close()

if __name__ == '__main__':
    # generate_list(sys.argv[1], sys.argv[2])
    generate_bundle(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))