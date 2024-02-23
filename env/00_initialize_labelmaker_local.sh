# exit when any command fails
set -e

# make sure submodules are updated
git submodule update --init --recursive

env_name=labelmaker
echo ${ENV_FOLDER}

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

# decide software version
pip install packaging
python ${ENV_FOLDER}/versions.py \
  --target_cuda_version ${target_cuda_version} \
  --target_torch_version ${target_torch_version} \
  --target_gcc_version ${target_gcc_version}

source ${ENV_FOLDER}/INSTALLED_VERSIONS.sh
echo ${INSTALLED_CUDA_VERSION}
echo ${INSTALLED_CUDA_ABBREV}
echo ${INSTALLED_PYTORCH_VERSION}
echo ${INSTALLED_GCC_VERSION}
echo ${INSTALLED_TORCHVISION_VERSION}
echo ${INSTALLED_OPEN3D_URL}

conda install -y -c conda-forge sysroot_linux-64=2.17 ffmpeg gxx=${INSTALLED_GCC_VERSION}
conda install -y -c "nvidia/label/cuda-${INSTALLED_CUDA_VERSION}" cuda
conda install -y -c anaconda openblas=0.3.20

conda deactivate
conda activate ${env_name}

conda_home="$(conda info | grep "active env location : " | cut -d ":" -f2-)"
conda_home="${conda_home#"${conda_home%%[![:space:]]*}"}"

echo ${conda_home}

which python
which pip
which nvcc
