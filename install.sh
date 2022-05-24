#conda create -n vibus python=3.8
#conda activate vibus
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/torch_stable.html -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pyparsing scipy ipykernel pyyaml plyfile
export CUDA_HOME=/usr/local/cuda-11.1/
cd MinkowskiEngine/ && python setup.py install && cd ..
cd pointnet2/ && python setup.py install && cd ..
