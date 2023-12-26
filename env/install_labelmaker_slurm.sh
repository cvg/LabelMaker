#!/usr/bin/bash
#SBATCH --job-name="labelmaker-env-build"
#SBATCH --output=labelmaker_env_build.out
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=64G
#SBATCH --tmp=32G

# exit when any command fails
set -e

# export ENV_FOLDER="$(pwd)/$(dirname "$0")"
export ENV_FOLDER=${HOME}/LabelMaker/env

# step 0: initiate environment
bash ${ENV_FOLDER}/00_initialize_labelmaker_slurm.sh
# step 1: install all pip packages
source ${ENV_FOLDER}/activate_labelmaker_slurm.sh && bash ${ENV_FOLDER}/01_pip_packages_install.sh

# step 2: install mask3d
source ${ENV_FOLDER}/activate_labelmaker_slurm.sh && bash ${ENV_FOLDER}/02.0_mask3d_detectron_2.sh
source ${ENV_FOLDER}/activate_labelmaker_slurm.sh && bash ${ENV_FOLDER}/02.1_mask3d_minkowskiengine.sh
source ${ENV_FOLDER}/activate_labelmaker_slurm.sh && bash ${ENV_FOLDER}/02.2_mask3d_others.sh

# step 3-7: omnidata+hha+cmx, grounded_sam, ovseg, internimage, labelmaker
source ${ENV_FOLDER}/activate_labelmaker_slurm.sh && bash ${ENV_FOLDER}/03_omnidata_hha_cmx.sh
source ${ENV_FOLDER}/activate_labelmaker_slurm.sh && bash ${ENV_FOLDER}/04_grounded_sam.sh
source ${ENV_FOLDER}/activate_labelmaker_slurm.sh && bash ${ENV_FOLDER}/05_ovseg.sh
source ${ENV_FOLDER}/activate_labelmaker_slurm.sh && bash ${ENV_FOLDER}/06_internimage.sh
source ${ENV_FOLDER}/activate_labelmaker_slurm.sh && bash ${ENV_FOLDER}/07_install_labelmaker.sh
