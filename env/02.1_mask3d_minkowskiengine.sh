set -e
echo ${ENV_FOLDER}

cd ${ENV_FOLDER}/../3rdparty/Mask3D/third_party
rm -rf MinkowskiEngine
git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine"
cd MinkowskiEngine
git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228

python setup.py install --force_cuda --blas=openblas
