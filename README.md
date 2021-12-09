# Specular-Uncertainty Fiele for 3D Scene Segmentation

## Install 

```shell
conda create -n sufield python=3.8
conda activate sufield

# pytorch=1.10.0 cudatoolkit=11.3.1 torchvision=0.11.1
conda install pytorch torchvision cudatoolkit -c pytorch
conda install scikit-learn scikit-learnex

pip install open3d
pip install plyfile tqdm pymeshlab potpourri3d
pip install pathos

conda install yapf

# CUDA <= 11.1
sudo apt install build-essential python3-dev libopenblas-dev
pip install ninja
pip install -U MinkowskiEngine --install-option="--blas=openblas" -v --no-deps
```