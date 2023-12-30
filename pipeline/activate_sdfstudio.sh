export PATH=/miniconda3/condabin:$PATH
env_name=sdfstudio
eval "$(conda shell.bash hook)"
conda activate $env_name
conda_home="$(conda info | grep "active env location : " | cut -d ":" -f2-)"
conda_home="${conda_home#"${conda_home%%[![:space:]]*}"}"

export CUDA_HOST_COMPILER="$conda_home/bin/gcc"
export CUDA_PATH="$conda_home"
export CUDA_HOME=$CUDA_PATH
export LD_LIBRARY_PATH=$conda_home/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH="$conda_home/lib/stubs:$LIBRARY_PATH"
export TCNN_CUDA_ARCHITECTURES=75
