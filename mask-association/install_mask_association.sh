env_name=sdfstudio
eval "$(conda shell.bash hook)"
conda activate $env_name

pip install igraph==0.11.8
pip install deeplake==3.9.31
pip install scipy==1.14.1
pip install kornia==0.6.12
pip install natsort
pip install Cython
pip install numpy==1.26.4
pip install scikit-learn==1.5.2 scipy==1.14.1 imageio==2.21.1 scikit-image==0.19.3

git clone https://github.com/cvg/Hierarchical-Localization.git

cd Hierarchical-Localization

pip install -e .

cd ../optimized

python setup.py build_ext --inplace