# this code includes running the whole labelmaker pipeline, including preprocessing from arkitscene to our custom format, run all individual models and then concensus and then 2D and 3D lifting.
# downloading is not included
# this bash file is not meant for modification into other dataset, needs further modification
# nor is it capable of tuning configuration, yet
env_name=labelmaker
eval "$(conda shell.bash hook)"
conda activate $env_name

conda_home="$(conda info | grep "active env location : " | cut -d ":" -f2-)"
conda_home="${conda_home#"${conda_home%%[![:space:]]*}"}"

echo $conda_home

which python
which pip
which nvcc

# add cuda compiler to path
export CUDA_HOST_COMPILER="$conda_home/bin/gcc"
export CUDA_PATH="$conda_home"
export CUDA_HOME=$CUDA_PATH
export LD_LIBRARY_PATH=$conda_home/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH="$conda_home/lib/stubs:$LIBRARY_PATH"
export TCNN_CUDA_ARCHITECTURES=75

if [ -z "$1" ]; then
  echo "No ARKitScene directory specified!"
  exit 1
else
  original_dir=$1
fi

if [ -z "$2" ]; then
  echo "No target directory specified!"
  exit 1
else
  target_dir=$2
fi

# preprocessing
python scripts/arkitscenes2labelmaker.py \
  --scan_dir ${original_dir} \
  --target_dir ${target_dir}

# extract mask3D
python models/mask3d_inst.py \
  --seed 42 \
  --workspace ${target_dir}

python models/mask3d_inst.py \
  --seed 43 \
  --output intermediate/scannet200_mask3d_2 \
  --workspace ${target_dir}

# extract omnidata normal
python models/omnidata_normal.py \
  --workspace ${target_dir}

python models/omnidata_depth.py \
  --workspace ${target_dir}

# extract hha depth, higher jobs may lead to failure
python models/hha_depth.py \
  --n_jobs 4 \
  --workspace ${target_dir}

# cmx
python models/cmx.py \
  --workspace ${target_dir}

python models/cmx.py --flip \
  --workspace ${target_dir}

# internimage
python models/internimage.py \
  --workspace ${target_dir}

python models/internimage.py --flip \
  --workspace ${target_dir}

# grounded sam
python models/grounded_sam.py \
  --workspace ${target_dir}

python models/grounded_sam.py --flip \
  --workspace ${target_dir}

# ovseg
python models/ovseg.py \
  --workspace ${target_dir}

python models/ovseg.py --flip \
  --workspace ${target_dir}

# consensus
python labelmaker/consensus.py \
  --workspace ${target_dir} --n_jobs 8

# point lifting
python labelmaker/lifting_3d/lifting_points.py \
  --workspace ${target_dir}

conda deactivate

# 3D lifting, mesh extracting, and rendering
bash labelmaker/lifting_3d/lifting.sh ${target_dir}
