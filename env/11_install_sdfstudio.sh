set -e
echo ${ENV_FOLDER}

# install open3d
pip install $INSTALLED_OPEN3D_URL

# install other packages
pip install gin-config pandas dask dask-jobqueue prefect prefect-dask

# remove open3d dependency
sed -i 's/"open3d>=0.16.0"/#"open3d>=0.16.0"/g' ${ENV_FOLDER}/../3rdparty/sdfstudio/pyproject.toml
# install sdfstudio
pip install ${ENV_FOLDER}/../3rdparty/sdfstudio
# ns-install-cli

# install labelmaker also
pip install -e ${ENV_FOLDER}/..

# install torch at this stage
pip install torch==$INSTALLED_PYTORCH_VERSION+$INSTALLED_CUDA_ABBREV torchvision==$INSTALLED_TORCHVISION_VERSION+$INSTALLED_CUDA_ABBREV --index-url https://download.pytorch.org/whl/$INSTALLED_CUDA_ABBREV
