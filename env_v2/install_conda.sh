env_name=labelmaker

conda env create -f env_v2/conda.yaml
# export CUDA_HOST_COMPILER="${HOME}/.conda/envs/${env_name}/bin/gcc"
# export CUDA_PATH="${HOME}/.conda/envs/${env_name}"
# export CUDA_HOME="${HOME}/.conda/envs/${env_name}"

PIP=$HOME/.conda/envs/${env_name}/bin/pip
$PIP install -r env_v2/00_download_sync.txt
$PIP install -r env_v2/01_preprocess.txt


