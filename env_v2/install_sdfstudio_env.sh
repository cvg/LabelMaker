# exit when any command fails
set -e

# make sure submodules are updated
git submodule update --init --recursive

env_name=sdfstudio
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

# create env, install gcc cuda and openblas
conda create --name $env_name --yes python=$target_python_version
eval "$(conda shell.bash hook)"
conda activate $env_name

pip install packaging
python $dir_name/versions.py --target_cuda_version $target_cuda_version --target_torch_version 1.12.1 --target_gcc_version 9.5.0

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
export LD_LIBRARY_PATH=$conda_home/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH="$conda_home/lib/stubs:$LIBRARY_PATH"
export TCNN_CUDA_ARCHITECTURES=75
export AM_I_DOCKER=1
export BUILD_WITH_CUDA=1
export FORCE_CUDA=1

# install open3d before sdfstudio
pip install $INSTALLED_OPEN3D_URL

# install other packages
pip install gin-config pandas

# remove open3d dependency
sed -i 's/"open3d>=0.16.0"/#"open3d>=0.16.0"/g' $dir_name/../3rdparty/sdfstudio/pyproject.toml

# install sdfstudio
pip install $dir_name/../3rdparty/sdfstudio
# ns-install-cli

# install labelmaker also
pip install -e $dir_name/..

pip install torch==$INSTALLED_PYTORCH_VERSION+$INSTALLED_CUDA_ABBREV torchvision==$INSTALLED_TORCHVISION_VERSION+$INSTALLED_CUDA_ABBREV --index-url https://download.pytorch.org/whl/$INSTALLED_CUDA_ABBREV

# install tcnn
conda install -y -c anaconda git
pip install "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
