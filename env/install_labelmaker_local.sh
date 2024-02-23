# exit when any command fails
set -e

# decide which version of python cuda pytorch torchvision to use
if [ -z "$1" ]; then
  target_python_version="3.9"
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

export ENV_FOLDER="$(pwd)/$(dirname "$0")"

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
