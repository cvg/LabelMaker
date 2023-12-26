echo ${ENV_FOLDER}

env_name=sdfstudio
eval "$(conda shell.bash hook)"
conda activate $env_name
conda_home="$(conda info | grep "active env location : " | cut -d ":" -f2-)"
conda_home="${conda_home#"${conda_home%%[![:space:]]*}"}"

export CUDA_HOST_COMPILER="$conda_home/bin/gcc"
export CUDA_PATH="$conda_home"
export CUDA_HOME=$CUDA_PATH
export TCNN_CUDA_ARCHITECTURES=75
export MAX_JOBS=6
export AM_I_DOCKER=1
export BUILD_WITH_CUDA=1
export FORCE_CUDA=1

# set software version in environment variable
source ${ENV_FOLDER}/INSTALLED_VERSIONS.sh
