# exit when any command fails
set -e

env_name=labelmaker
dir_name="$(pwd)/$(dirname "$0")"

echo $dir_name

eval "$(conda shell.bash hook)"
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

# specify NLTK download location
export NLTK_DATA="$dir_name/../3rdparty/nltk_data"

# testing
rm -rf $dir_name/../testing/test_scan/intermediate
cd $dir_name/../testing/test_models
pytest test_cmx_00_omnidata.py
pytest test_cmx_01_hha.py
pytest test_cmx_02_cmx.py
pytest test_grounded_sam.py
pytest test_internimage.py
pytest test_mask3d.py
pytest test_ovseg.py
pytest test_omnidata_normal.py
pytest test_consensus.py
