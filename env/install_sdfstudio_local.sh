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

export ENV_FOLDER="$(pwd)/$(dirname "$0")"

bash ${ENV_FOLDER}/10_initialize_sdfstudio_local.sh ${target_python_version} ${target_cuda_version}
source ${ENV_FOLDER}/activate_sdfstudio_local.sh && bash ${ENV_FOLDER}/11_pip_packages_install.sh
source ${ENV_FOLDER}/activate_sdfstudio_local.sh && bash ${ENV_FOLDER}/12_install_tcnn.sh
