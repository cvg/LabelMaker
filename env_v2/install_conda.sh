env_name=labelmaker
dir_name="$(pwd)/$(dirname "$0")"

# create env, install gcc cuda and openblas
conda env create -f $dir_name/conda.yaml

# add cuda compiler to path
export CUDA_HOST_COMPILER="${HOME}/.conda/envs/$env_name/bin/gcc"
export CUDA_PATH="${HOME}/.conda/envs/$env_name"
export CUDA_HOME=$CUDA_PATH

PYTHON="${HOME}/.conda/envs/$env_name/bin/python"
PIP="${HOME}/.conda/envs/$env_name/bin/pip"

# install download and preprocessing
$PIP install -r "$dir_name/00_download_sync.txt"
$PIP install -r "$dir_name/01_preprocess.txt"

# install torch
$PIP install -r "$dir_name/02_torch.txt"

# install mask3d dependency
$PIP install -r "$dir_name/03_mask3d.txt" # torch-scatter and detectron2
## pointnet2
cd $dir_name/../3rdparty/Mask3D/third_party/pointnet2
$PYTHON setup.py install
## minkowskiengine
cd $dir_name/../3rdparty/MinkowskiEngine
git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
$PYTHON setup.py install --force_cuda --blas=openblas 
## scannet segmentor
cd $dir_name/../3rdparty/ScanNet/Segmentator
git checkout 3e5726500896748521a6ceb81271b0f5b2c0e7d2
make
## install mask3d package
cd $dir_name/../3rdparty/Mask3D
$PYTHON setup.py install

# install ovseg
$PIP install -r "$dir_name/04_ovseg.txt"

# install HHA
$PIP install -r "$dir_name/05_hha_depth.txt"
cd $dir_name/../mmseg/Depth2HHA-python
$PYTHON setup.py install

# install grounded sam
$PIP install $dir_name/../3rdparty/recognize-anything/
$PIP install $dir_name/../3rdparty/Grounded-Segment-Anything/segment_anything
$PIP install $dir_name/../3rdparty/Grounded-Segment-Anything/GroundingDINO

# install cmx
$PIP install -U openmim
MIM="${HOME}/.conda/envs/$env_name/bin/mim"
$MIM install mmengine
$MIM install "mmcv>=2.0.0" 
cd $dir_name/../3rdparty/mmsegmentation
$PIP install -v -e .
