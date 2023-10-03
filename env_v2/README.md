On euler, to use headless rendering version of open3d, build OSMesa according to https://github.com/quantaji/open3d-manylinux2014/blob/main/osmesa_euler_build_install.sh, add llvm module in euler, and add OSMesa into LD_LIBRARY_PATH
```sh
export LD_LIBRARY_PATH=${HOME}/osmesa/lib:$LD_LIBRARY_PATH
```
