# Specular-Uncertainty Fiele for 3D Scene Segmentation

## Install 

```shell
conda create -n sufield python=3.8
conda activate sufield

# pytorch=1.10.0 cudatoolkit=11.3.1 torchvision=0.11.1
conda install pytorch torchvision cudatoolkit -c pytorch
# conda install pytorch=1.7.1 torchvision cudatoolkit=11.0 -c pytorch
conda install scikit-learn 
pip install scikit-learn-intelex

pip install open3d plyfile tqdm pymeshlab potpourri3d pathos
conda install yapf

# CUDA <= 11.1, set CUDA_HOME env
sudo apt install build-essential python3-dev libopenblas-dev
pip install ninja
pip install -U MinkowskiEngine --install-option="--blas=openblas" -v --no-deps
# pointnet2
cd pointnet2
python setup.py install

```