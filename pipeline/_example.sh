#!/usr/bin/bash
#SBATCH --job-name="labelmaker"
#SBATCH --output=%j.out
#SBATCH --time=0:30:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --tmp=32G

module purge
module load eth_proxy

export PATH="/cluster/project/cvg/labelmaker/miniconda3/bin:${PATH}"

env_name=labelmaker
env_dir=/cluster/project/cvg/labelmaker/${env_name}_venv
export PATH=${env_dir}/bin:${PATH}

env_name=labelmaker
eval "$(conda shell.bash hook)"
conda activate $env_name
conda_home="$(conda info | grep "active env location : " | cut -d ":" -f2-)"
conda_home="${conda_home#"${conda_home%%[![:space:]]*}"}"

export AM_I_DOCKER=1
export BUILD_WITH_CUDA=1
export CUDA_HOST_COMPILER="$conda_home/bin/gcc"
export CUDA_PATH="$conda_home"
export CUDA_HOME=$CUDA_PATH
export FORCE_CUDA=1
export MAX_JOBS=6
export NLTK_DATA="${ENV_FOLDER}/../3rdparty/nltk_data"
export TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6"

# start postgresql server
# pg_ctl -D /cluster/home/guanji/prefect_postgresql/ -l logfile start
# prefect config set PREFECT_API_DATABASE_CONNECTION_URL="postgresql+asyncpg://postgres:12345@localhost:5432/prefect"

# srun --ntasks=1 prefect server start &

# sleep 60

# prefect config set PREFECT_HOME="${TMPDIR}/prefect"
# export PREFECT_HOME="${TMPDIR}/prefect"
# prefect server database reset -y

# prefect config view --show-sources

which python
# prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
# prefect config set PREFECT_API_URL=http://0.0.0.0:4200/api
python pipeline/_example.py

wait
