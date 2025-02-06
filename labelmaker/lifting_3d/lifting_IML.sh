# exit when any command fails
set -e

dir_name="$(pwd)/$(dirname "$0")"

repo_dir="$dir_name/../.."
echo "repo_dir is $repo_dir"

# activate environment
env_name=sdfstudio
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

wandb online

# get scene folder
if [ -z "$1" ]; then
  echo "Workspace directory not specified!!!"
  exit 1
else
  WORKSPACE=$1
fi
# WORKSPACE="/lustre/scratch/data/s6yczhan_hpc-thesis/labelmaker_data/scene0518_00"
echo "Workspace is $WORKSPACE"

export TCNN_CUDA_ARCHITECTURES=75

# receive the subsample from the second argument
if [ -z "$2" ]; then
  echo "subsample not specified, use default subsample=1"
  subsample=1
else
  subsample=$2
fi

# preprocessing
python "$repo_dir"/labelmaker/lifting_3d/preprocessing_IML.py \
  --sampling $subsample \
  --workspace $WORKSPACE

# # train
method=IML
# experiment_name=IML—0518
# Extract the last part of the WORKSPACE path and prepend "IML"
experiment_name="IML-"$(basename "$WORKSPACE")
echo "experiment is $experiment_name"
output_dir=${WORKSPACE}/intermediate/${experiment_name}
preprocess_data_dir=${WORKSPACE}/intermediate/IML_preprocessing

export WANDB_MODE=online
wandb online

# about 26G gpu memory, 1207.58s
# currently semantic loss is switched of (semantic-loss-mult 0.0, include-semantics False)m no mono prior (normal, depth) is used (include-mono-prior False)
ns-train ${method} \
  --experiment-name $experiment_name \
  --pipeline.model.semantic-loss-mult 1 \
  --pipeline.model.instance-loss-mult 1 \
  --pipeline.model.semantic-patch-loss-mult 0.00 \
  --pipeline.model.semantic-patch-loss-min-step 1000 \
  --pipeline.model.semantic-ignore-label 0 \
  --pipeline.model.semantic-num-layers 4 \
  --pipeline.model.semantic-layer-width 512 \
  --pipeline.model.instance-num-layers 4 \
  --pipeline.model.instance-layer-width 512 \
  --trainer.steps-per-eval-image 1000 \
  --trainer.steps-per-eval-all-images 100000 \
  --trainer.steps-per-save 10000 \
  --trainer.max-num-iterations 30000 \
  --pipeline.datamanager.train-num-rays-per-batch 2048 \
  --output-dir ${WORKSPACE}/intermediate \
  --vis wandb \
  IML-data \
  --data ${preprocess_data_dir} \
  --include-semantics True \
  --include-auxiliary-semantics True \
  --include-instances True

# the job below may OOM sometimes, so we wait such that all GPU memory is free
# sleep 60

# locate results
results_dir=${output_dir}/$(ls $output_dir)
train_id=$(ls $results_dir)

config=$results_dir/$train_id/config.yml

# # extract mesh
# ns-extract-mesh \
#   --load-config $config \
#   --create-visibility-mask True \
#   --output-path $results_dir/$train_id/mesh_visible.ply \
#   --resolution 512
# # # sleep 60

# render class labels
render_dir=${WORKSPACE}/nerfacto_lifted_semantic
mkdir -p $render_dir
ns-render --camera-path-filename $preprocess_data_dir/camera_path.json \
  --traj filename \
  --output-format images \
  --rendered-output-names semantics \
  --output-path $render_dir \
  --load-config $config


# render instance labels
render_dir=${WORKSPACE}/nerfacto_lifted_instance
mkdir -p $render_dir
ns-render --camera-path-filename $preprocess_data_dir/camera_path.json \
  --traj filename \
  --output-format images \
  --rendered-output-names instances \
  --output-path $render_dir \
  --load-config $config
