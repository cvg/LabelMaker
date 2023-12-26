echo ${ENV_FOLDER}

# load module
module load gcc/8.2.0 cuda/11.3.1 python/3.9.9 ffmpeg/3.2.4 openblas/0.3.20

# activate python
env_name=labelmaker
env_dir=${HOME}/${env_name}_venv
source ${env_dir}/bin/activate

# add cuda, compiler path, extra for  compilation
export AM_I_DOCKER=1
export BUILD_WITH_CUDA=1
export CUDA_HOST_COMPILER=$(which gcc)
export CUDA_PATH=${CUDA_HOME}
export FORCE_CUDA=1
export MAX_JOBS=6 # euler user is only 2 core...
export TCNN_CUDA_ARCHITECTURES=75

# set software version in environment variable
source ${ENV_FOLDER}/INSTALLED_VERSIONS.sh
