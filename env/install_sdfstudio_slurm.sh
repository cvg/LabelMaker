#!/usr/bin/bash
#SBATCH --job-name="sdfstudio-env-build"
#SBATCH --output=sdfstudio_env_build.out
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=64G
#SBATCH --tmp=32G

# exit when any command fails
set -e

# export ENV_FOLDER="$(pwd)/$(dirname "$0")"
export ENV_FOLDER=${HOME}/LabelMaker/env

# also setting pip cache directory to avoid storage quota limit
PIP_CACHE_DIR=${TMPDIR}/pip_cache
mkdir -p ${PIP_CACHE_DIR}

bash ${ENV_FOLDER}/10_initialize_sdfstudio_slurm.sh
source ${ENV_FOLDER}/activate_sdfstudio_slurm.sh && bash ${ENV_FOLDER}/11_install_sdfstudio.sh
source ${ENV_FOLDER}/activate_sdfstudio_slurm.sh && bash ${ENV_FOLDER}/12_install_tcnn.sh
