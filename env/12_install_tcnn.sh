set -e
echo ${ENV_FOLDER}

pip install "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
