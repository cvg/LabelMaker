# ! NOTE: on euler, I fixed python=3.10, cuda=11.3, torch=1.12.0
# ! I choose to install the python venv in $HOME/labelmaker_venv and $HOME/sdfstudio_venv

# exit when any command fails
set -e

# this ENV_FOLDER is labelmaker_repo/env
echo ${ENV_FOLDER}

module load gcc/8.2.0 cuda/11.3.1 python/3.10.4 openblas/0.3.20

target_python_version="3.10"
target_cuda_version="11.3"
target_torch_version="1.12.1"
target_gcc_version="9.5.0"

# make sure submodules are updated
git submodule update --init --recursive

# create virtual environment
env_name=sdfstudio
env_dir=/cluster/project/cvg/labelmaker/${env_name}_venv
rm -rf ${env_dir}
python -m venv ${env_dir}

# activate
source ${env_dir}/bin/activate

# decide software version
pip install packaging
python $ENV_FOLDER/versions.py \
  --target_cuda_version $target_cuda_version \
  --target_torch_version $target_torch_version \
  --target_gcc_version $target_gcc_version

source $ENV_FOLDER/INSTALLED_VERSIONS.sh
echo $INSTALLED_CUDA_VERSION
echo $INSTALLED_CUDA_ABBREV
echo $INSTALLED_PYTORCH_VERSION
echo $INSTALLED_GCC_VERSION
echo $INSTALLED_TORCHVISION_VERSION
echo $INSTALLED_OPEN3D_URL

which python
which pip
which nvcc
