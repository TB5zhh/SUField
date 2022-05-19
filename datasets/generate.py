import os
import sys


def generate(scannet_path, save_dir):
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
    generate(sys.argv[1], sys.argv[2])