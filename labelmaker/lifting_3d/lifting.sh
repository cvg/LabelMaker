# exit when any command fails
set -e

dir_name="$(pwd)/$(dirname "$0")"
repo_dir="$dir_name/../.."

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
echo "Workspace is $WORKSPACE"

export TCNN_CUDA_ARCHITECTURES=75

# preprocessing
python "$repo_dir"/labelmaker/lifting_3d/preprocessing.py \
  --sampling 1 \
  --workspace $WORKSPACE

# # train
method=neus-facto
experiment_name=sdfstudio_train
output_dir=${WORKSPACE}/intermediate/${experiment_name}
preprocess_data_dir=${WORKSPACE}/intermediate/sdfstudio_preprocessing

export WANDB_MODE=online
wandb online

# about 26G gpu memory, 1207.58s
# currently semantic loss is switched of (semantic-loss-mult 0.0, include-semantics False)m no mono prior (normal, depth) is used (include-mono-prior False)
ns-train ${method} \
  --experiment-name $experiment_name \
  --pipeline.model.sdf-field.use-grid-feature True \
  --pipeline.model.sdf-field.hidden-dim 256 \
  --pipeline.model.sdf-field.num-layers 2 \
  --pipeline.model.sdf-field.num-layers-color 2 \
  --pipeline.model.sdf-field.semantic-num-layers 4 \
  --pipeline.model.sdf-field.semantic_layer_width 512 \
  --pipeline.model.sdf-field.use-appearance-embedding False \
  --pipeline.model.sdf-field.geometric-init True \
  --pipeline.model.sdf-field.inside-outside True \
  --pipeline.model.sdf-field.bias 0.8 \
  --pipeline.model.sdf-field.beta-init 0.3 \
  --pipeline.model.sensor-depth-l1-loss-mult 10.0 \
  --pipeline.model.sensor-depth-sdf-loss-mult 6000.0 \
  --pipeline.model.sensor-depth-freespace-loss-mult 10.0 \
  --pipeline.model.sensor-depth-truncation 0.015 \
  --pipeline.model.mono-normal-loss-mult 0.02 \
  --pipeline.model.mono-depth-loss-mult 0.00 \
  --pipeline.model.semantic-loss-mult 0.1 \
  --pipeline.model.semantic-patch-loss-mult 0.00 \
  --pipeline.model.semantic-patch-loss-min-step 1000 \
  --pipeline.model.semantic-ignore-label 0 \
  --trainer.steps-per-eval-image 1000 \
  --trainer.steps-per-eval-all-images 100000 \
  --trainer.steps-per-save 10000 \
  --trainer.max-num-iterations 20001 \
  --pipeline.datamanager.train-num-rays-per-batch 2048 \
  --pipeline.model.eikonal-loss-mult 0.1 \
  --pipeline.model.background-model none \
  --output-dir ${WORKSPACE}/intermediate \
  --vis wandb \
  sdfstudio-data \
  --data ${preprocess_data_dir} \
  --include-sensor-depth True \
  --include-semantics True \
  --include-mono-prior True

# the job below may OOM sometimes, so we wait such that all GPU memory is free
# sleep 60

# locate results
results_dir=${output_dir}/$(ls $output_dir)
train_id=$(ls $results_dir)

config=$results_dir/$train_id/config.yml

# extract mesh
ns-extract-mesh \
  --load-config $config \
  --create-visibility-mask True \
  --output-path $results_dir/$train_id/mesh_visible.ply \
  --resolution 512
# # sleep 60

# render class labels
render_dir=${WORKSPACE}/neus_lifted
mkdir -p $render_dir
ns-render --camera-path-filename $preprocess_data_dir/camera_path.json \
  --traj filename \
  --output-format images \
  --rendered-output-names semantics \
  --output-path $render_dir \
  --load-config $config
