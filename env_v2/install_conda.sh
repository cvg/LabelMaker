env_name=labelmaker
dir_name="$(pwd)/$(dirname "$0")"

# create env, install gcc cuda and openblas
conda create --name $name --yes python=3.10
conda install -y -c "conda-forge" gxx=11.4.0
conda install -y -c "nvidia/label/cuda-11.8.0" cuda
# install openblas
conda install -y -c anaconda openblas=0.3.20

# add cuda compiler to path
export CUDA_HOST_COMPILER="${HOME}/.conda/envs/$env_name/bin/gcc"
export CUDA_PATH="${HOME}/.conda/envs/$env_name"
export CUDA_HOME=$CUDA_PATH
export TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6"

# install download and preprocessing
pip install -r "$dir_name/00_download_sync.txt"
pip install -r "$dir_name/01_preprocess.txt"

# install torch
pip install -r "$dir_name/02_torch.txt"

install mask3d dependency
pip install -r "$dir_name/03_mask3d.txt" # torch-scatter and detectron2
# minkowskiengine
cd $dir_name/../3rdparty/Mask3D/third_party
rm -rf MinkowskiEngine
git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine"
cd MinkowskiEngine
git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
python setup.py install --force_cuda --blas=openblas
# scannet segmentor
cd $dir_name/../3rdparty/Mask3D/third_party
rm -rf ScanNet
git clone https://github.com/ScanNet/ScanNet.git
cd ScanNet/Segmentator
git checkout 3e5726500896748521a6ceb81271b0f5b2c0e7d2
make
## pointnet2
cd $dir_name/../3rdparty/Mask3D/third_party/pointnet2
python setup.py install
## install mask3d package
cd $dir_name/../3rdparty/Mask3D
python setup.py install

# install HHA
pip install -r "$dir_name/05_hha_depth.txt"
cd $dir_name/../mmseg/Depth2HHA-python
python setup.py install

# create folder for omnidata
mkdir -p $dir_name/..3rdparty/omnidata/omnidata_tools/torch/pretrained_models/
pip install -r "$dir_name/07_omnidata.txt"

# install grounded sam
pip install -r "$dir_name/06_grounded_sam.txt"
pip install $dir_name/../3rdparty/recognize-anything/
pip install $dir_name/../3rdparty/Grounded-Segment-Anything/segment_anything
pip install $dir_name/../3rdparty/Grounded-Segment-Anything/GroundingDINO

# install cmx
pip install -r "$dir_name/08_cmx.txt"
cd $dir_name/../3rdparty/mmsegmentation
pip install -v -e .

# install ovseg, ovseg customize clip, so reinstall from this after grounded sam
pip install -r "$dir_name/04_ovseg.txt"
cd $dir_name/../3rdparty/ov-seg/third_party/CLIP
python -m pip install -Ue .
export NLTK_DATA="$dir_name/../3rdparty/nltk_data"
mkdir -p NLTK_DATA
python -m nltk.downloader -d $NLTK_DATA wordnet

# internimage
cd $dir_name/../3rdparty/InternImage/segmentation/ops_dcnv3
sh ./make.sh
