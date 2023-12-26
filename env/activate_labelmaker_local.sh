echo ${ENV_FOLDER}

env_name=labelmaker
eval "$(conda shell.bash hook)"
conda activate $env_name
conda_home="$(conda info | grep "active env location : " | cut -d ":" -f2-)"
conda_home="${conda_home#"${conda_home%%[![:space:]]*}"}"

export CUDA_HOST_COMPILER="$conda_home/bin/gcc"
export CUDA_PATH="$conda_home"
export CUDA_HOME=$CUDA_PATH
export TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6"
export MAX_JOBS=6
export AM_I_DOCKER=1
export BUILD_WITH_CUDA=1
export FORCE_CUDA=1
export NLTK_DATA="${ENV_FOLDER}/../3rdparty/nltk_data"

# set software version in environment variable
source ${ENV_FOLDER}/INSTALLED_VERSIONS.sh
