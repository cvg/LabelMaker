On euler, to use headless rendering version of open3d, build OSMesa according to https://github.com/quantaji/open3d-manylinux2014/blob/main/osmesa_euler_build_install.sh, add llvm module in euler, and add OSMesa into LD_LIBRARY_PATH
```sh
export LD_LIBRARY_PATH=${HOME}/osmesa/lib:$LD_LIBRARY_PATH
```


## Install Gounded SAM
First update all submodule
```sh
git submodule update --init --recursive
```

Install pytorch and other required packages
```sh
pip install -r ./env_v2/02_grounded_sam.txt
```
Install RAM
```sh
pip install ./3rdparty/recognize-anything/
```

Install SAM
```sh
pip install ./3rdparty/Grounded-Segment-Anything/segment_anything
```

Install Grounding DINO
```sh
export CUDA_HOST_COMPILER="${HOME}/.conda/envs/labelmaker/bin/gcc"
export CUDA_PATH="${HOME}/.conda/envs/labelmaker"
export CUDA_HOME=$CUDA_PATH
pip install ./3rdparty/Grounded-Segment-Anything/GroundingDINO
```
