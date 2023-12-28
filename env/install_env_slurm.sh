#!/usr/bin/bash
#SBATCH --job-name="labelmaker-env-build"
#SBATCH --output=labelmaker_env_build.out
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=4G
#SBATCH --tmp=32G

set -e

module purge
module load eth_proxy

export ENV_FOLDER=${HOME}/LabelMaker/env
export PIP_CACHE_DIR=${TMPDIR}/pip_cache
export PATH="/cluster/project/cvg/labelmaker/miniconda3/bin:${PATH}"

target_python_version="3.9"
target_cuda_version="11.3"
target_torch_version="1.12.0"
target_gcc_version="9.5.0"

# step 0: initiate environment
bash ${ENV_FOLDER}/00_initialize_labelmaker_local.sh \
  ${target_python_version} \
  ${target_cuda_version} \
  ${target_torch_version} \
  ${target_gcc_version}

# step 1: install all pip packages
source ${ENV_FOLDER}/activate_labelmaker_local.sh && bash ${ENV_FOLDER}/01_pip_packages_install.sh

# step 2: install mask3d
source ${ENV_FOLDER}/activate_labelmaker_local.sh && bash ${ENV_FOLDER}/02.0_mask3d_detectron_2.sh
source ${ENV_FOLDER}/activate_labelmaker_local.sh && bash ${ENV_FOLDER}/02.1_mask3d_minkowskiengine.sh
source ${ENV_FOLDER}/activate_labelmaker_local.sh && bash ${ENV_FOLDER}/02.2_mask3d_others.sh

# step 3-7: omnidata+hha+cmx, grounded_sam, ovseg, internimage, labelmaker
source ${ENV_FOLDER}/activate_labelmaker_local.sh && bash ${ENV_FOLDER}/03_omnidata_hha_cmx.sh
source ${ENV_FOLDER}/activate_labelmaker_local.sh && bash ${ENV_FOLDER}/04_grounded_sam.sh
source ${ENV_FOLDER}/activate_labelmaker_local.sh && bash ${ENV_FOLDER}/05_ovseg.sh
source ${ENV_FOLDER}/activate_labelmaker_local.sh && bash ${ENV_FOLDER}/06_internimage.sh
source ${ENV_FOLDER}/activate_labelmaker_local.sh && bash ${ENV_FOLDER}/07_install_labelmaker.sh

target_python_version="3.10"
target_cuda_version="11.3"

bash ${ENV_FOLDER}/10_initialize_sdfstudio_local.sh ${target_python_version} ${target_cuda_version}
source ${ENV_FOLDER}/activate_sdfstudio_local.sh && bash ${ENV_FOLDER}/11_install_sdfstudio.sh
source ${ENV_FOLDER}/activate_sdfstudio_local.sh && bash ${ENV_FOLDER}/12_install_tcnn.sh
