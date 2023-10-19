env_name=labelmaker
dir_name="$(pwd)/$(dirname "$0")"

# create env, install gcc cuda and openblas
conda create --name $env_name --yes python=3.10
eval "$(conda shell.bash hook)"
conda activate $env_name

conda install -y -c "conda-forge" gxx=11.4.0
conda install -y -c "nvidia/label/cuda-11.8.0" cuda
conda install -y -c anaconda openblas=0.3.20

conda deactivate
conda activate $env_name

which python
which pip
which nvcc

# add cuda compiler to path
export CUDA_HOST_COMPILER="${HOME}/.conda/envs/$env_name/bin/gcc"
export CUDA_PATH="${HOME}/.conda/envs/$env_name"
export CUDA_HOME=$CUDA_PATH
export TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6"
export MAX_JOBS=6

# specify NLTK download location
export NLTK_DATA="$dir_name/../3rdparty/nltk_data"
mkdir -p NLTK_DATA

# install all dependency from pypi
pip install -r "$dir_name/requirements.txt"

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
cd $dir_name/../mmseg/Depth2HHA-python
pip install .
# Step 3: install cmx
cd $dir_name/../3rdparty/mmsegmentation
pip install -v -e .

# install grounded sam
pip install $dir_name/../3rdparty/recognize-anything/
pip install $dir_name/../3rdparty/Grounded-Segment-Anything/segment_anything
pip install $dir_name/../3rdparty/Grounded-Segment-Anything/GroundingDINO

# install ovseg, ovseg customize clip, so reinstall from this after grounded sam
cd $dir_name/../3rdparty/ov-seg/third_party/CLIP
python -m pip install -Ue .
python -m nltk.downloader -d $NLTK_DATA wordnet

# install internimage
cd $dir_name/../3rdparty/InternImage/segmentation/ops_dcnv3
sh ./make.sh
