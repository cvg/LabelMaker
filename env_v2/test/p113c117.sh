name=p113c117
conda create --name $name --yes python=3.10
conda activate $name

# install cuda, gcc, this has to be the first one otherwise extremely slow
conda install -y -c "conda-forge" gxx=11.4.0
conda install -y -c "nvidia/label/cuda-11.7.0" cuda
# install openblas
conda install -y -c anaconda openblas=0.3.20

export CUDA_HOST_COMPILER="${HOME}/.conda/envs/$name/bin/gcc"
export CUDA_PATH="${HOME}/.conda/envs/$name"
export CUDA_HOME=$CUDA_PATH

# install pytorch
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# Mask3D
# torch-scatter
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.1%2Bcu117.html
# detectron2
pip3 install 'git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf' --no-deps
# install pointnet2
cd $HOME/Projects/LabelMaker/3rdparty/Mask3D/third_party/pointnet2
python setup.py install
# install minkowskiengine
cd $HOME/Projects/LabelMaker/3rdparty/MinkowskiEngine
git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
python setup.py install --force_cuda --blas=openblas
# install scannet segmentator
cd $HOME/Projects/LabelMaker/3rdparty/ScanNet/Segmentator
git checkout 3e5726500896748521a6ceb81271b0f5b2c0e7d2
make
# mask3d
cd $HOME/Projects/LabelMaker/3rdparty/Mask3D
python setup.py install

# install ram and grounding dino
pip install fairscale
cd $HOME/Projects/LabelMaker
pip install ./3rdparty/recognize-anything/
pip install ./3rdparty/Grounded-Segment-Anything/segment_anything
pip install ./3rdparty/Grounded-Segment-Anything/GroundingDINO
