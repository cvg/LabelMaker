set -e
echo ${ENV_FOLDER}

# install scannet segmentor
cd ${ENV_FOLDER}/../3rdparty/Mask3D/third_party
rm -rf ScanNet
git clone https://github.com/ScanNet/ScanNet.git
cd ScanNet/Segmentator
git checkout 3e5726500896748521a6ceb81271b0f5b2c0e7d2
make

# pointnet2
cd ${ENV_FOLDER}/../3rdparty/Mask3D/third_party/pointnet2
python setup.py install

# install mask3d package
cd ${ENV_FOLDER}/../3rdparty/Mask3D
pip install .
pip install --no-deps --force-reinstall --upgrade omegaconf==2.2.0 hydra-core==1.0.5
