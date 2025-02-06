# exit when any command fails
set -e

# make sure submodules are updated
git submodule update --init --recursive

env_name=labelmaker
dir_name="$(pwd)/$(dirname "$0")"

echo $dir_name

# decide which version of python cuda pytorch torchvision to use
if [ -z "$1" ]; then
  target_python_version="3.10"
else
  target_python_version=$1
fi

if [ -z "$2" ]; then
  target_cuda_version="unset"
else
  target_cuda_version=$2
fi

if [ -z "$3" ]; then
  target_torch_version="unset"
else
  target_torch_version=$3
fi

if [ -z "$4" ]; then
  target_gcc_version="unset"
else
  target_gcc_version=$4
fi

# create env, install gcc cuda and openblas
conda create --name $env_name --yes python=$target_python_version
eval "$(conda shell.bash hook)"
conda activate $env_name

pip install packaging
python $dir_name/versions.py --target_cuda_version $target_cuda_version --target_torch_version $target_torch_version --target_gcc_version $target_gcc_version

source $dir_name/INSTALLED_VERSIONS.sh
echo $INSTALLED_CUDA_VERSION
echo $INSTALLED_CUDA_ABBREV
echo $INSTALLED_PYTORCH_VERSION
echo $INSTALLED_GCC_VERSION
echo $INSTALLED_TORCHVISION_VERSION
echo $INSTALLED_OPEN3D_URL

conda install -y -c "conda-forge" gxx=$INSTALLED_GCC_VERSION
conda install -y -c conda-forge sysroot_linux-64=2.17
conda install -y -c "nvidia/label/cuda-$INSTALLED_CUDA_VERSION" cuda
conda install -y -c anaconda openblas=0.3.20

conda deactivate
conda activate $env_name

conda_home="$(conda info | grep "active env location : " | cut -d ":" -f2-)"
conda_home="${conda_home#"${conda_home%%[![:space:]]*}"}"

echo $conda_home

which python
which pip
which nvcc

# add cuda compiler to path
export CUDA_HOST_COMPILER="$conda_home/bin/gcc"
export CUDA_PATH="$conda_home"
export CUDA_HOME=$CUDA_PATH
export TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6"
export MAX_JOBS=6
export AM_I_DOCKER=1
export BUILD_WITH_CUDA=1
export FORCE_CUDA=1 

# specify NLTK download location
export NLTK_DATA="$dir_name/../3rdparty/nltk_data"
mkdir -p $NLTK_DATA

# TODO add git checkout of all repository to keep version consistent

# install all dependency from pypi
pip install -r "$dir_name/requirements.txt"

# install open3d
pip install $INSTALLED_OPEN3D_URL

# install torch and torch-scater, they are cuda-version dependent
# Pytorch
pip install torch==$INSTALLED_PYTORCH_VERSION+$INSTALLED_CUDA_ABBREV torchvision==$INSTALLED_TORCHVISION_VERSION+$INSTALLED_CUDA_ABBREV --index-url https://download.pytorch.org/whl/$INSTALLED_CUDA_ABBREV
# torch-scatter
pip install torch-scatter --index-url "" -f "https://data.pyg.org/whl/torch-${INSTALLED_PYTORCH_VERSION}%2B${INSTALLED_CUDA_ABBREV}.html"
pip install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/${INSTALLED_CUDA_ABBREV}/torch${INSTALLED_PYTORCH_VERSION}/index.html

# install mask3d
# Step 1: install detectron 2 and minkowskiengine
pip install "git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf"
cd $dir_name/../3rdparty/Mask3D/third_party
rm -rf MinkowskiEngine
git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine"
cd MinkowskiEngine
git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
python setup.py install --force_cuda --blas=openblas
# Step 2: install scannet segmentor
cd $dir_name/../3rdparty/Mask3D/third_party
rm -rf ScanNet
git clone https://github.com/ScanNet/ScanNet.git
cd ScanNet/Segmentator
git checkout 3e5726500896748521a6ceb81271b0f5b2c0e7d2
make
## Step 3: pointnet2
cd $dir_name/../3rdparty/Mask3D/third_party/pointnet2
python setup.py install
## Step 4: install mask3d package
cd $dir_name/../3rdparty/Mask3D
pip install .
pip install --no-deps --force-reinstall --upgrade omegaconf==2.2.0 hydra-core==1.0.5

# install omnidata + hha + cmx
# Step 1: create folder and install omnidata # might be deprecated as weight will be stored at other path
mkdir -p $dir_name/../3rdparty/omnidata/omnidata_tools/torch/pretrained_models/
# Step 2: install HHA
cd $dir_name/../3rdparty/Depth2HHA-python
pip install .
# Step 3: install cmx
cd $dir_name/../3rdparty/mmsegmentation
pip install -v -e .
# Step 4: create an empty txt for cmx eval configuration
cd $dir_name/../3rdparty/RGBX_Semantic_Segmentation
touch empty.txt
# Step 5: replace collectioin.iterable into collection.abc.iterable
sed -i 's/collections.Iterable/collections.abc.Iterable/g' $dir_name/../3rdparty/RGBX_Semantic_Segmentation/utils/transforms.py

# install grounded sam
pip install $dir_name/../3rdparty/recognize-anything/
pip install $dir_name/../3rdparty/Grounded-Segment-Anything/segment_anything
pip install $dir_name/../3rdparty/Grounded-Segment-Anything/GroundingDINO

# install ovseg, ovseg customize clip, so reinstall from this after grounded sam
cd $dir_name/../3rdparty/ov-seg/third_party/CLIP
python -m pip install -e .
python -m nltk.downloader -d $NLTK_DATA wordnet

# install internimage
# # avoid an error when no cuda runtime available
sed -i 's/torch.cuda.is_available()/True/g' $dir_name/../3rdparty/InternImage/segmentation/ops_dcnv3/setup.py
cd $dir_name/../3rdparty/InternImage/segmentation/ops_dcnv3
sh ./make.sh

# install labelmaker
pip install -e $dir_name/../.

pip uninstall clip
