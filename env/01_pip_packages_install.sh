set -e

which python
which pip
which nvcc

# install all dependency from pypi
pip install -r "${ENV_FOLDER}/requirements.txt"

# install open3d
pip install ${INSTALLED_OPEN3D_URL}

# install torch and torch-scater, they are cuda-version dependent
# Pytorch
pip install torch==${INSTALLED_PYTORCH_VERSION}+${INSTALLED_CUDA_ABBREV} torchvision==${INSTALLED_TORCHVISION_VERSION}+${INSTALLED_CUDA_ABBREV} --index-url https://download.pytorch.org/whl/${INSTALLED_CUDA_ABBREV}

# torch-scatter
pip install torch-scatter --index-url "" -f "https://data.pyg.org/whl/torch-${INSTALLED_PYTORCH_VERSION}%2B${INSTALLED_CUDA_ABBREV}.html"

# mmcv
pip install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/${INSTALLED_CUDA_ABBREV}/torch${INSTALLED_PYTORCH_VERSION}/index.html
